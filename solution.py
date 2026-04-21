from __future__ import annotations

from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):
    """
    Applicant template.

    Replace every method body with your own implementation while preserving the
    public API used by the official tests.

    The intended contract is:
    - one writer and one or more readers
    - shared state visible across processes
    - bounded storage with reusable space after readers advance
    - reads and writes report how many bytes are actually available
    """

    _NO_READER = -1

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
    ):
        """
        Open or create the shared buffer.

        Expected behavior:
        - validate constructor arguments
        - allocate or attach to shared memory
        - initialize any shared metadata needed to track writer and reader state
        - set up local views/fields used by the rest of the methods

        Parameters:
        - `name`: shared memory block name
        - `create`: `True` for the creator/owner, `False` to attach to an existing block
        - `size`: logical payload capacity in bytes
        - `num_readers`: number of reader slots to support
        - `reader`: reader index for this instance, or `_NO_READER` for the writer instance
        - `cache_align` / `cache_size`: optional metadata-layout knobs; you may ignore
          them internally as long as validation and behavior remain correct
        """
        if size <= 0:
            raise ValueError(f"SharedBuffer '{name}': size must be > 0")
        if num_readers < 1:
            raise ValueError(f"SharedBuffer '{name}': num_readers must be >= 1")
        if cache_align and (cache_size <= 0 or cache_size & (cache_size - 1)):
            raise ValueError("cache_size must be a positive power of two")
        if reader != self._NO_READER and not (0 <= reader < num_readers):
            raise ValueError(f"reader index {reader} out of range for num_readers={num_readers}")

        STATIC_FIELDS = 3
        READER_FIELDS = 3
        UINT64 = 8

        raw_header_bytes = UINT64 * (STATIC_FIELDS + num_readers * READER_FIELDS)

        if cache_align:
            raw_header_bytes = (raw_header_bytes + cache_size - 1) & ~(cache_size - 1)

        self.header_size = raw_header_bytes
        self.ring_buffer_size = size
        total_size = self.header_size + self.ring_buffer_size

        super().__init__(name=name, create=create, size=total_size)

        num_header_u64 = STATIC_FIELDS + num_readers * READER_FIELDS
        self.header = np.ndarray(
            (num_header_u64,),
            dtype=np.uint64,
            buffer=self.buf,
            offset=0,
        )

        if create:
            self.header[:] = 0
            self.header[0] = np.uint64(self.ring_buffer_size)
            self.header[2] = np.uint64(num_readers)

        self.cache_align = cache_align
        self.cache_size = cache_size
        self.num_readers = num_readers
        self.reader = reader

        self.ring_buffer = memoryview(self.buf)[self.header_size : self.header_size + self.ring_buffer_size]

        self._size_idx = 0
        self._write_pos_idx = 1
        self._num_readers_idx = 2

        self._STATIC = STATIC_FIELDS
        self._READER_FIELDS = READER_FIELDS

        if self.reader == self._NO_READER:
            self.reader_pos_index = None
        else:
            self.reader_pos_index = self._STATIC + self.reader * self._READER_FIELDS

        self.write_pos = int(self.header[self._write_pos_idx])
        self.reader_pos = 0 if self.reader_pos_index is not None else None

    def close(self) -> None:
        """
        Release local views and close this process's handle to the shared memory.

        This should not destroy the buffer for other attached processes.
        """
        ring_buffer = getattr(self, "ring_buffer", None)
        if ring_buffer is not None:
            try:
                ring_buffer.release()
            except Exception:
                pass
            self.ring_buffer = None

        header = getattr(self, "header", None)
        if header is not None:
            self.header = None

        try:
            super().close()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        """
        Enter the context manager.

        Reader instances are expected to mark themselves active while inside the
        context. Writer-only instances can simply return `self`.
        """
        if self.reader != self._NO_READER:
            self.set_reader_active(True)
        return self

    def __exit__(self, *_):
        """
        Exit the context manager.

        Reader instances are expected to mark themselves inactive on exit, then
        close local resources.
        """
        if self.reader != self._NO_READER:
            self.set_reader_active(False)
        self.close()

    def calculate_pressure(self) -> int:
        """
        Return current writer pressure as an integer percentage.

        Pressure is based on how much of the bounded storage is currently in use
        relative to the slowest active reader.
        """
        max_amount_writable = self.compute_max_amount_writable(force_rescan=True)
        used = self.ring_buffer_size - max_amount_writable
        pressure = int((used / self.ring_buffer_size) * 100)
        return pressure

    def int_to_pos(self, value: int) -> int:
        """
        Convert an absolute position counter into a position inside the bounded payload area.

        If your design does not use modulo arithmetic internally, you may still
        keep this helper as the mapping from logical positions to buffer offsets.
        """
        return value % self.ring_buffer_size

    def update_reader_pos(self, new_reader_pos: int) -> None:
        """
        Store this reader's absolute read position in shared state.

        This must fail clearly when called on a writer-only instance.
        """
        if self.reader_pos_index is None:
            raise RuntimeError("update_reader_pos called on a writer-only instance")
        self.header[self.reader_pos_index] = np.uint64(new_reader_pos)
        self.reader_pos = new_reader_pos

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        if self.reader_pos_index is None:
            raise RuntimeError("set_reader_active called on a writer-only instance")
        self.header[self.reader_pos_index + 1] = np.uint64(1 if active else 0)

    def is_reader_active(self) -> bool:
        """
        Return whether this reader is currently marked active.

        This must fail clearly when called on a writer-only instance.
        """
        if self.reader_pos_index is None: 
            raise RuntimeError(f'writer only instance: no reader is currently active')
        return self.header[self.reader_pos_index + 1] == 1
    
    def update_write_pos(self, new_writer_pos: int) -> None:
        """
        Store the writer's absolute write position in shared state.

        The write position is what makes newly written bytes visible to readers.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.update_write_pos")

    def inc_writer_pos(self, inc_amount: int) -> None:
        """
        Advance the writer's absolute position by `inc_amount` bytes.

        This is how a writer publishes bytes after copying them into the buffer.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.inc_writer_pos")

    def inc_reader_pos(self, inc_amount: int) -> None:
        """
        Advance this reader's absolute position by `inc_amount` bytes.

        This is how a reader consumes bytes after reading them.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.inc_reader_pos")

    def get_write_pos(self) -> int:
        """
        Return the current absolute writer position.

        Readers can use this to resynchronize or compute how much data is available.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.get_write_pos")

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """
        Return how many bytes the writer can safely expose right now.

        This should take active readers into account. `force_rescan=True` is used
        by the tests to ensure externally updated reader positions are observed.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.compute_max_amount_writable")

    def jump_to_writer(self) -> None:
        """
        Move this reader directly to the current writer position.

        Use this when a reader has fallen too far behind and old unread data is
        no longer retained.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.jump_to_writer")

    def expose_writer_mem_view(self, size: int) -> RingView:
        """
        Return a writable view tuple for up to `size` bytes.

        The return shape is:
        - `mv1`: first writable view
        - `mv2`: optional second writable view if the exposed region is split
        - `actual_size`: how many bytes are actually writable right now
        - `split`: whether the writable region is split across two views

        If less than `size` bytes are currently writable, clamp to the amount
        available rather than raising.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.expose_writer_mem_view")

    def expose_reader_mem_view(self, size: int) -> RingView:
        """
        Return a readable view tuple for up to `size` bytes.

        The shape matches `expose_writer_mem_view()`. If less than `size` bytes
        are currently readable, clamp to the amount available rather than raising.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.expose_reader_mem_view")

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """
        Copy bytes from `src` into the exposed writer view(s).

        If `src` is larger than the destination region, copy only the prefix that fits.
        This helper should not publish data by itself; publishing happens when the
        writer position is advanced.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.simple_write")

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy bytes from the exposed reader view(s) into `dst`.

        If `dst` is smaller than the readable region, copy only the prefix that fits.
        This helper should not consume data by itself; consumption happens when the
        reader position is advanced.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.simple_read")

    def write_array(self, arr: np.ndarray) -> int:
        """
        Write a NumPy array's raw bytes into the shared buffer.

        Return the number of bytes written. If the full array does not fit, the
        contract used by the tests expects this method to return `0`.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.write_array")

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        raise NotImplementedError("TODO: implement SharedBuffer.read_array")
