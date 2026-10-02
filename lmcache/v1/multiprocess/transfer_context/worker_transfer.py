# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Callable, Protocol, cast
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import (
    EngineDrivenKernelGroupLayout,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    engine_group_layer_indices,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    ChunkBuffer,
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    engine_driven_chunk_shape,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import get_device_spec, resolve_kv_wrapper_factory
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.kv_wrap import wrap_kv_caches

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


# Helper functions
def _supports_async_primitives() -> bool:
    """Probe whether the worker device supports the async store primitives.

    The async engine-driven store path needs a stream, an event exposing
    ``record``/``synchronize``/``wait``, and pinned (page-locked) host memory.
    When any of these is unavailable (e.g. a CPU-only backend), the factory
    falls back to the synchronous :class:`EngineDrivenTransferContext`. This
    dispatch is internal and capability-based; there is no user-facing
    async/sync flag.

    Returns:
        True if all required async primitives are available, else False.
    """
    if not hasattr(torch_dev, "Stream") or not hasattr(torch_dev, "Event"):
        return False
    # CPU-only stub exposes Stream/Event but has no real async capability.
    if hasattr(torch_dev, "is_available") and not torch_dev.is_available():
        return False
    try:
        stream = torch_dev.Stream()
        event = torch_dev.Event()
    except Exception:
        return False
    for attr in ("record", "synchronize", "wait"):
        if not callable(getattr(event, attr, None)):
            del stream, event
            return False
    del stream, event
    try:
        probe = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        del probe
    except (RuntimeError, TypeError):
        return False
    return True


def _build_engine_driven_context(
    instance_id: int,
    req_client: RequestClient,
) -> "TransferContext":
    """Build the engine-driven context, async when device-capable else sync.

    Routes the ``ENGINE_DRIVEN`` and AUTO branches through a single capability
    check. ``AsyncEngineDrivenTransferContext`` is imported lazily to avoid an
    import cycle and to keep the synchronous path free of stream/event
    dependencies.

    Returns:
        ``AsyncEngineDrivenTransferContext`` when async primitives are
        available, otherwise ``EngineDrivenTransferContext``.
    """
    if _supports_async_primitives():
        # First Party
        from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
            AsyncEngineDrivenTransferContext,
        )

        logger.info("Using AsyncEngineDrivenTransferContext for store path")
        return AsyncEngineDrivenTransferContext(instance_id, req_client)

    logger.info("Using EngineDrivenTransferContext (sync) for store path")
    return EngineDrivenTransferContext(instance_id, req_client)


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode %r (valid: %s)" % (raw, valid)
        ) from exc


def _build_lmcache_driven_context(
    device_type: str,
    instance_id: int,
    req_client: RequestClient,
) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        resolve_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext(instance_id, req_client)


class IPCEvent(Protocol):
    """Protocol for device events used by transport operations."""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (async ordering primitive)."""


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _get_kv_device(kv_caches: dict[str, torch.Tensor]) -> torch.device:
    """Return the device shared by a non-empty KV-cache mapping.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name.

    Returns:
        The device of the first KV-cache tensor.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    if not kv_caches:
        raise ValueError("LMCache-driven transfer requires at least one KV cache")
    return get_device(next(iter(kv_caches.values())))


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. Device-handle paths
    return event-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Bind this context to a single worker and request client.

        Args:
            instance_id: Worker process instance identifier used by all
                context-owned transport requests.
            req_client: Transport-neutral client used by this context. The
                adapter retains ownership and closes it after the context.
        """
        self._instance_id = instance_id
        self._req_client = req_client
        self._registration_request_sent = False
        self._closed = False
        self._lifecycle_lock = threading.Lock()

    def _submit_registration(
        self, submit: Callable[[], MessagingFuture[Any]]
    ) -> MessagingFuture[Any]:
        """Submit REGISTER before allowing a concurrent UNREGISTER.

        If shutdown won the race before the request was sent, registration is
        rejected. Once submission begins, unregistration remains available even
        if waiting for the registration response times out.
        """
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Transfer context is closed.")
            self._registration_request_sent = True
            return submit()

    def _submit_unregistration(
        self, submit: Callable[[], MessagingFuture[Any]]
    ) -> MessagingFuture[Any] | None:
        """Submit UNREGISTER only after this context sent REGISTER."""
        with self._lifecycle_lock:
            if not self._registration_request_sent:
                return None
            return submit()

    def _is_closed(self) -> bool:
        """Return whether adapter shutdown closed this context."""
        with self._lifecycle_lock:
            return self._closed

    def _mark_closed(self) -> None:
        """Prevent future registration requests from a racing caller."""
        with self._lifecycle_lock:
            self._closed = True

    @abstractmethod
    def register(
        self,
        _kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_timeout: Timeout in seconds for synchronous request wait.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            engine_type: Serving engine that produced the caches. Only
                consumed by the handle path; adapters should pass their
                own :class:`EngineType` so this transport stays engine-
                neutral. Defaults to :attr:`EngineType.VLLM` for
                backwards compatibility.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    @abstractmethod
    def unregister(self) -> MessagingFuture[Any] | None:
        """Start unregistering this context's server-side KV-cache state.

        Concrete contexts select the unregister RPC that matches the protocol
        used by :meth:`register`. The returned future lets the caller apply
        its own lifecycle timeout policy before :meth:`close` releases local
        state.

        Returns:
            A future that resolves when the server acknowledges unregistration,
            or ``None`` when no registration request was sent.

        """

    def register_q(
        self,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register the paged Q ring with the server under the same worker
        instance ID but different model_name (model_name##query).

        Args:
            q_caches: Worker Q cache tensors keyed by layer name.
            model_name: Model name used by cache keys (model_name##query).
            world_size: KV world size.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.
            mq_timeout: Timeout in seconds for synchronous request wait.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (now only lmcache-driven).
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """
        raise NotImplementedError(
            "Q ring registration is not supported by this transfer context"
        )

    @abstractmethod
    def create_recorded_event(self) -> IPCEvent | None:
        """Create the event needed to order the next transfer.

        Returns:
            A recorded device event when the transfer context needs stream
            ordering, or ``None`` when the context orders transfers
            synchronously without an event.

        Raises:
            RuntimeError: If the context has not been registered or cannot
                create the event required by its transfer protocol.
        """

    def submit_q_store(
        self,
        request_id: str,
        key: Any,
        q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a Q ring store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key for the Q store range (query-specific model_name).
            q_caches: Q ring tensors keyed by layer name.
            block_ids: Q ring block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (only the lmcache-driven path does).
            RuntimeError: If register_q() was not called first.
        """
        raise NotImplementedError(
            "Q ring store is not supported by this transfer context"
        )

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object, or ``None`` when the concrete
                context does not require one.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object, or ``None`` when the concrete
                context does not require one.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""

    @abstractmethod
    def flush_inflight_stores(self) -> None:
        """Synchronize any in-flight gather operations.

        Subclasses must implement this method. Contexts with no deferred
        operations should implement it as a no-op. Async contexts that
        defer GPU->CPU gather work must block until all in-flight stores
        have completed, so that vLLM cannot overwrite paged KV blocks
        before they are read.
        """


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (accelerator IPC,
    or SHM wrappers for CPU with IPC-like semantics) and the LMCache server
    performs direct device-side data transfer.
    """

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Initialize a handle-path context bound to one worker.

        Args:
            instance_id: Worker process instance identifier.
            req_client: Transport client for this worker.
        """
        super().__init__(instance_id, req_client)
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None
        self._mq_timeout: float = 0.0
        self._inflight_stores: list[MessagingFuture] = []
        self._inflight_lock = threading.Lock()

    @staticmethod
    def _store_settled(future: MessagingFuture) -> bool:
        """Whether the server is done with this store's engine KV blocks.

        ``query()`` raises when the store's RPC failed, so a failed store is
        reported as settled: it is no longer reading the blocks, and its error
        is surfaced by the request path that owns it rather than here.

        Args:
            future: A store future returned by ``submit_store``.

        Returns:
            True if the store completed or failed, False if still in flight.
        """
        try:
            return future.query()
        except Exception:
            logger.debug("Treating a failed store as settled", exc_info=True)
            return True

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            mq_timeout: Timeout for the registration response.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata.
            engine_type: Serving engine that produced the caches.

        Raises:
            RuntimeError: If event IPC is unsupported for the KV-cache device.
            ValueError: If ``kv_caches`` is empty.
        """
        device = _get_kv_device(kv_caches)
        event_backend = get_event_ipc_backend(device)
        event_backend.check_event_support(device)

        future = self._submit_registration(
            lambda: self._req_client.register_kv_cache(
                self._instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                engine_type,
                layout_hints or {},
                list(engine_group_infos),
            )
        )
        future.result(timeout=mq_timeout)
        if self._is_closed():
            return
        self._device = device
        self._event_backend = event_backend
        self._mq_timeout = mq_timeout

    def create_recorded_event(self) -> IPCEvent:
        """Create and record an exportable event for handle-based transfer.

        Returns:
            An interprocess-capable event recorded on the current stream.

        Raises:
            RuntimeError: If :meth:`register` has not completed.
        """
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before creating transfer events."
            )
        event = self._event_backend.create_event(self._device)
        self._event_backend.record_event(event, torch_dev.current_stream())
        return cast(IPCEvent, event)

    def unregister(self) -> MessagingFuture[Any] | None:
        """Start handle-path unregistration for this worker instance.

        Returns:
            A future for the server's unregister acknowledgement.

        """
        return self._submit_unregistration(
            lambda: self._req_client.unregister_kv_cache(self._instance_id)
        )

    def register_q(
        self,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        future = self._req_client.register_q_cache(
            self._instance_id,
            wrap_kv_caches(q_caches),
            model_name,
            world_size,
            EngineType.VLLM,
            layout_hints or {},
            list(engine_group_infos),
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a handle-based store ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the store range.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders reads of the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        if event is None:
            raise RuntimeError("LMCache-driven transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        future = self._req_client.store(
            key, self._instance_id, block_ids, event_ipc_handle
        ).to_device_future(
            device=self._device,
            event_backend=self._event_backend,
        )
        with self._inflight_lock:
            self._inflight_stores = [
                f for f in self._inflight_stores if not self._store_settled(f)
            ]
            self._inflight_stores.append(future)
        return future

    def submit_q_store(
        self,
        _request_id: str,
        key: Any,
        _q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_q_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._req_client.store_q(
            key, self._instance_id, block_ids, event_ipc_handle
        ).to_device_future(
            device=self._device,
            event_backend=self._event_backend,
        )

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a handle-based retrieve ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the retrieve range.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders writes to the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).
            skip_first_n_tokens: Initial tokens the server must not overwrite.

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if self._device is None or self._event_backend is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        if event is None:
            raise RuntimeError("LMCache-driven transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._req_client.retrieve(
            key,
            self._instance_id,
            block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        ).to_device_future(
            device=self._device,
            event_backend=self._event_backend,
        )

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        self._mark_closed()
        self._device = None
        self._event_backend = None

    def flush_inflight_stores(self) -> None:
        """Block until the server has finished reading the engine KV blocks.

        In this mode the server copies the blocks on its own stream after the
        forward pass that produced them, and the engine never waits for that
        copy.  When the scheduler preempts a request it frees those blocks
        immediately and may hand them to another request in the same step,
        whose forward pass would overwrite blocks the server is still reading
        and commit the wrong KV under the preempted request's keys.

        A timeout is logged rather than raised, so a slow server degrades to a
        possibly stale store instead of a crashed engine.
        """
        with self._inflight_lock:
            pending = [f for f in self._inflight_stores if not self._store_settled(f)]
            self._inflight_stores = []
        for future in pending:
            try:
                if not future.wait(timeout=self._mq_timeout):
                    logger.warning(
                        "A store did not finish within %.1fs; its KV blocks may "
                        "be overwritten while the server is still reading them",
                        self._mq_timeout,
                    )
            except Exception:
                logger.exception("Failed waiting for an in-flight store")


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    """

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Initialize an engine-driven context bound to one worker.

        Args:
            instance_id: Worker process instance identifier.
            req_client: Transport client for this worker.
        """
        super().__init__(instance_id, req_client)
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        # Set by register() for a hybrid model: its kernel groups, with each
        # group's layer names, engine blocks per chunk, and chunk tensor shape.
        self._kv_groups: KVLayerGroupsManager | None = None
        self._group_layer_names: list[list[str]] = []
        self._group_blocks_per_chunk: list[int] = []
        self._group_chunk_shapes: list[torch.Size] = []

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/pickle context created by ``register``.

        Raises:
            RuntimeError: If accessed before ``register`` has run.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "EngineDrivenTransferContext is not registered, call register() first."
            )
        return self._engine_driven_context

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the non-GPU context server.

        A hybrid model (more than one engine KV cache group) is split into
        the kernel groups the LMCache-driven path builds, and each chunk is
        stored as one object holding one tensor per kernel group.
        ``engine_type`` is accepted to satisfy the base interface.
        """
        del engine_type  # unused on the engine-driven path
        self._layout_hints = layout_hints
        if len(engine_group_infos) > 1:
            group_layouts = self._build_kernel_groups(
                kv_caches, blocks_in_chunk, layout_hints, engine_group_infos
            )
            first = group_layouts[0]
            block_size = (
                cast(KVLayerGroupsManager, self._kv_groups)
                .kernel_groups[0]
                .slots_per_block
            )
            use_mla_flag = first.use_mla
            layout_desc = MemoryLayoutDesc(
                shapes=self._group_chunk_shapes,
                dtypes=[getattr(torch, group.dtype_str) for group in group_layouts],
            )
            payload = RegisterEngineDrivenContextPayload(
                instance_id=self._instance_id,
                model_name=model_name,
                world_size=world_size,
                block_size=block_size,
                num_layers=first.num_layers,
                hidden_dim_size=first.hidden_dim_size,
                dtype_str=first.dtype_str,
                use_mla=use_mla_flag,
                num_physical_slots=first.num_physical_slots,
                kernel_groups=group_layouts,
            )
        else:
            # TODO: per-group compression (EngineGroupInfo.tokens_per_block vs
            # the tensor-detected slot count, e.g. DeepSeek V4) is only handled
            # on the CUDA path. The non-CUDA path is yet to be implemented.
            (
                block_size,
                num_layers,
                hidden_dim_size,
                dtype_str,
                engine_kv_format,
                kv_size,
            ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
            self._engine_kv_format = engine_kv_format

            # The wire field is named use_mla but only drives the object plane
            # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
            use_mla_flag = kv_size == 1
            num_physical_slots = blocks_in_chunk * block_size
            layout_desc = MemoryLayoutDesc(
                shapes=[
                    engine_driven_chunk_shape(
                        num_layers, num_physical_slots, hidden_dim_size, use_mla_flag
                    )
                ],
                dtypes=[getattr(torch, dtype_str)],
            )
            payload = RegisterEngineDrivenContextPayload(
                instance_id=self._instance_id,
                model_name=model_name,
                world_size=world_size,
                block_size=block_size,
                num_layers=num_layers,
                hidden_dim_size=hidden_dim_size,
                dtype_str=dtype_str,
                use_mla=use_mla_flag,
                num_physical_slots=num_physical_slots,
            )

        future = self._submit_registration(
            lambda: self._req_client.register_kv_cache_engine_driven_context(payload)
        )
        response = future.result(timeout=mq_timeout)
        if self._is_closed():
            return
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            self._req_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s)",
            self._instance_id,
            supported_transfer_mode,
        )

    def _build_kernel_groups(
        self,
        kv_caches: dict[str, torch.Tensor],
        blocks_in_chunk: int,
        layout_hints: LayoutHints | None,
        engine_group_infos: Sequence[EngineGroupInfo],
    ) -> list[EngineDrivenKernelGroupLayout]:
        """Split a hybrid model's layers into kernel groups.

        Groups are built exactly as the LMCache-driven path builds them, so
        stored objects share one layout across transfer modes.

        Returns:
            Each kernel group's part of a chunk object, in kernel-group order.

        Raises:
            NotImplementedError: If the engine groups report different (or no)
                block sizes, or a kernel group is compressed.
        """
        block_sizes = {info.tokens_per_block for info in engine_group_infos}
        if len(block_sizes) != 1 or 0 in block_sizes:
            raise NotImplementedError(
                "Engine-driven transfer needs one reported block size across "
                f"KV cache groups, got {sorted(block_sizes)}"
            )
        chunk_tokens = blocks_in_chunk * block_sizes.pop()
        normalized, engine_kv_formats = normalize_and_discover_per_layer_formats(
            list(kv_caches.values()),
            engine_group_layer_indices(engine_group_infos),
            EngineType.VLLM,
            layout_hints,
        )
        kv_groups = KVLayerGroupsManager(
            list(normalized),
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=chunk_tokens,
        )

        layer_names = list(kv_caches)
        group_layouts: list[EngineDrivenKernelGroupLayout] = []
        self._group_layer_names = []
        self._group_blocks_per_chunk = []
        self._group_chunk_shapes = []
        for group_idx, group in enumerate(kv_groups.kernel_groups):
            if group.slots_per_block != group.tokens_per_block:
                raise NotImplementedError(
                    f"Kernel group {group_idx} is compressed (tokens_per_block="
                    f"{group.tokens_per_block}, slots_per_block="
                    f"{group.slots_per_block}); engine-driven transfer does "
                    "not support compressed KV groups"
                )
            layout = EngineDrivenKernelGroupLayout(
                num_layers=group.num_layers,
                num_physical_slots=group.calculate_slots(chunk_tokens),
                hidden_dim_size=group.hidden_dim_size,
                dtype_str=str(group.dtype).replace("torch.", ""),
                use_mla=group.shape_desc.kv_size == 1,
            )
            group_layouts.append(layout)
            self._group_layer_names.append(
                [layer_names[idx] for idx in group.layer_indices]
            )
            self._group_blocks_per_chunk.append(
                kv_groups.calculate_num_blocks(group_idx, chunk_tokens)
            )
            self._group_chunk_shapes.append(
                engine_driven_chunk_shape(
                    layout.num_layers,
                    layout.num_physical_slots,
                    layout.hidden_dim_size,
                    layout.use_mla,
                )
            )
        self._kv_groups = kv_groups
        logger.info(
            "Engine-driven transfer stores %d kernel groups per chunk object",
            len(group_layouts),
        )
        return group_layouts

    def _num_group_chunks(self, block_ids: list[list[int]]) -> int:
        """Return the chunk count shared by every kernel group's block IDs.

        Raises:
            ValueError: If there is not one block-ID list per kernel group, or
                the lists cover different numbers of chunks.
        """
        if len(block_ids) != len(self._group_blocks_per_chunk):
            raise ValueError(
                f"Expected block IDs for {len(self._group_blocks_per_chunk)} "
                f"kernel groups, got {len(block_ids)}"
            )
        num_chunks = {
            len(group_block_ids) // blocks_per_chunk
            for group_block_ids, blocks_per_chunk in zip(
                block_ids, self._group_blocks_per_chunk, strict=True
            )
        }
        if len(num_chunks) != 1:
            raise ValueError(
                f"Kernel groups' block IDs cover different chunk counts: {num_chunks}"
            )
        return num_chunks.pop()

    def _check_group_buffers(
        self, buffers: list[ChunkBuffer]
    ) -> list[list[torch.Tensor]]:
        """Return ``buffers`` as per-group parts, checked against the layout.

        Raises:
            ValueError: If a buffer is not one tensor per kernel group with the
                registered shape; the transfer kernels write through raw
                pointers, so a mismatch must never reach them.
        """
        for buffer in buffers:
            if (
                not isinstance(buffer, list)
                or [part.shape for part in buffer] != self._group_chunk_shapes
            ):
                raise ValueError(
                    "Chunk buffer does not match the registered kernel-group "
                    f"layout {[tuple(s) for s in self._group_chunk_shapes]}"
                )
        return cast(list[list[torch.Tensor]], buffers)

    def _gather_kernel_groups(
        self,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        out_buffers: list[ChunkBuffer] | None,
        chunk_indices: list[int] | None,
    ) -> list[list[torch.Tensor]]:
        """Gather each kernel group into its part of every chunk object."""
        self._num_group_chunks(block_ids)
        parts_out = (
            None if out_buffers is None else self._check_group_buffers(out_buffers)
        )
        kv_groups = cast(KVLayerGroupsManager, self._kv_groups)
        per_group: list[list[torch.Tensor]] = []
        for group_idx, group in enumerate(kv_groups.kernel_groups):
            per_group.append(
                gather_paged_kv_to_cpu(
                    {
                        name: kv_caches[name]
                        for name in self._group_layer_names[group_idx]
                    },
                    block_ids[group_idx],
                    self._group_blocks_per_chunk[group_idx],
                    layout_hints=self._layout_hints,
                    engine_kv_format=group.engine_kv_format,
                    out=(
                        None
                        if parts_out is None
                        else [parts[group_idx] for parts in parts_out]
                    ),
                    chunk_indices=chunk_indices,
                )
            )
        return [list(parts) for parts in zip(*per_group, strict=True)]

    def _scatter_kernel_groups(
        self,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        src_buffers: list[ChunkBuffer],
        skip_first_n_tokens: int,
    ) -> None:
        """Scatter each kernel group's part of every chunk object back."""
        self._num_group_chunks(block_ids)
        parts_in = self._check_group_buffers(src_buffers)
        kv_groups = cast(KVLayerGroupsManager, self._kv_groups)
        for group_idx, group in enumerate(kv_groups.kernel_groups):
            scatter_cpu_to_paged_kv(
                {name: kv_caches[name] for name in self._group_layer_names[group_idx]},
                block_ids[group_idx],
                [parts[group_idx] for parts in parts_in],
                self._group_blocks_per_chunk[group_idx],
                skip_first_n_tokens=skip_first_n_tokens,
                layout_hints=self._layout_hints,
                engine_kv_format=group.engine_kv_format,
            )

    def unregister(self) -> MessagingFuture[Any] | None:
        """Start engine-driven unregistration for this worker instance.

        Returns:
            A future for the server's unregister acknowledgement.

        """
        return self._submit_unregistration(
            lambda: self._req_client.unregister_kv_cache_engine_driven_context(
                self._instance_id
            )
        )

    def create_recorded_event(self) -> IPCEvent | None:
        """Return no event for the synchronous engine-driven transfer path.

        Returns:
            ``None`` because store and retrieve synchronize the active device
            before accessing or releasing KV-cache buffers.

        Raises:
            RuntimeError: If :meth:`register` has not completed.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before creating transfer events."
            )
        return None

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent | None,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, self._instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        cpu_chunks: list[ChunkBuffer]
        if self._kv_groups is not None:
            cpu_chunks = list(
                self._gather_kernel_groups(
                    kv_caches, block_ids, out_buffers, chunk_indices
                )
            )
        else:
            cpu_chunks = list(
                gather_paged_kv_to_cpu(
                    kv_caches,
                    _single_group_block_ids(block_ids),
                    blocks_in_chunk,
                    layout_hints=self._layout_hints,
                    engine_kv_format=self._engine_kv_format,
                    out=cast(list[torch.Tensor] | None, out_buffers),
                    chunk_indices=chunk_indices,
                )
            )
        # Gather issues async device->CPU copies on BOTH transports: into the
        # SHM slots when out_buffers is given, otherwise into fresh buffers that
        # commit_store serializes immediately. Either way the copies must be
        # complete first, so this is unconditional -- guarding it on out_buffers
        # left the pickle path serializing a buffer still being written.
        torch_dev.synchronize()
        ok = self._engine_driven_context.commit_store(
            key, self._instance_id, cpu_chunks
        )

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent | None,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        src_buffers = self._engine_driven_context.prepare_retrieve(
            key, self._instance_id
        )
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                if self._kv_groups is not None:
                    self._scatter_kernel_groups(
                        kv_caches, block_ids, src_buffers, skip_first_n_tokens
                    )
                else:
                    scatter_cpu_to_paged_kv(
                        kv_caches,
                        _single_group_block_ids(block_ids),
                        cast(list[torch.Tensor], src_buffers),
                        blocks_in_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        engine_kv_format=self._engine_kv_format,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, self._instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        self._mark_closed()
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def flush_inflight_stores(self) -> None:
        pass


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    *,
    instance_id: int,
    req_client: RequestClient,
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        instance_id: Worker process instance identifier bound to the context.
        req_client: Transport client bound to the context. The caller retains
            ownership and must close it after the context.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {get_device(v).type for v in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type, instance_id, req_client)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return _build_engine_driven_context(instance_id, req_client)
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext(instance_id, req_client)
    return _build_engine_driven_context(instance_id, req_client)
