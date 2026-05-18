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
        self.buffer_size = size
        total_size = self.header_size + self.buffer_size

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
            self.header[0] = np.uint64(self.buffer_size)
            self.header[2] = np.uint64(num_readers)

        self.cache_align = cache_align
        self.cache_size = cache_size
        self.num_readers = num_readers
        self.reader = reader

        self.buffer = self.buf[self.header_size : self.header_size + self.buffer_size]

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

        # Cached Reader Information 
        self._cached_reader_positions = [0] * num_readers
        self._cached_reader_active = [False] * num_readers

        if not create:
            self._rescan_readers()

        self._read_buf = bytearray(size)  

    def close(self) -> None:
        """
        Release local views and close this process's handle to the shared memory.

        This should not destroy the buffer for other attached processes.
        """
        ring_buffer = getattr(self, "buffer", None)
        if ring_buffer is not None:
            try:
                ring_buffer.release()
            except Exception:
                pass
            self.buffer = None

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
        used = self.buffer_size - max_amount_writable
        pressure = int((used / self.buffer_size) * 100)
        return pressure

    def int_to_pos(self, value: int) -> int:
        """
        Convert an absolute position counter into a position inside the bounded payload area.

        If your design does not use modulo arithmetic internally, you may still
        keep this helper as the mapping from logical positions to buffer offsets.
        """
        return value % self.buffer_size

    def update_reader_pos(self, new_reader_pos: int) -> None:
        """
        Store this reader's absolute read position in shared state.

        This must fail clearly when called on a writer-only instance.
        """
        if self.reader_pos_index is None:
            raise RuntimeError("update_reader_pos called on a writer-only instance")
        
        self.header[self.reader_pos_index] = new_reader_pos
        self.reader_pos = new_reader_pos
        self._cached_reader_positions[self.reader] = new_reader_pos

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        if self.reader_pos_index is None:
            raise RuntimeError("set_reader_active called on a writer-only instance")
        self.header[self.reader_pos_index + 1] = np.uint64(1 if active else 0)
        self._cached_reader_active[self.reader] = active


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
        self.header[self._write_pos_idx] = new_writer_pos
        self.write_pos = new_writer_pos

    def inc_writer_pos(self, inc_amount: int) -> None:
        """
        Advance the writer's absolute position by `inc_amount` bytes.

        This is how a writer publishes bytes after copying them into the buffer.
        """
        new_writer_pos = self.write_pos + inc_amount
        self.update_write_pos(new_writer_pos)

    def inc_reader_pos(self, inc_amount: int) -> None:
        """
        Advance this reader's absolute position by `inc_amount` bytes.

        This is how a reader consumes bytes after reading them.
        """
        if self.reader_pos_index is None:
            raise RuntimeError("expose_reader_mem_view called on a writer-only instance")
        new_reader_pos = self.reader_pos + inc_amount
        self.update_reader_pos(new_reader_pos)

    def get_write_pos(self) -> int:
        """
        Return the current absolute writer position.

        Readers can use this to resynchronize or compute how much data is available.
        """
        return int(self.header[self._write_pos_idx])

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """
        Return how many bytes the writer can safely expose right now.

        This should take active readers into account. `force_rescan=True` is used
        by the tests to ensure externally updated reader positions are observed.
        """
        if force_rescan:
            self._rescan_readers()
        slowest = self._cached_slowest_pos
        if slowest is None:
            return self.buffer_size
        used = self.write_pos - slowest
        return self.buffer_size - used

    def jump_to_writer(self) -> None:
        """
        Move this reader directly to the current writer position.

        Use this when a reader has fallen too far behind and old unread data is
        no longer retained.
        """
        self.update_reader_pos(self.get_write_pos())

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
        max_writable = self.compute_max_amount_writable()

        if max_writable < size: 
            max_writable = self.compute_max_amount_writable(force_rescan=True)

        actual_size = min(size, max_writable)

        write_offset = self.int_to_pos(self.write_pos)

        if write_offset + actual_size <= self.buffer_size:
            mv1 = self.buffer[write_offset : write_offset + actual_size]
            mv2 = None
            split = False
        else: 
            mv1 = self.buffer[write_offset:]
            mv2 = self.buffer[0: actual_size - len(mv1)]
            split = True 

        return (mv1, mv2, actual_size, split)


    def expose_reader_mem_view(self, size: int) -> RingView:
        """
        Return a readable view tuple for up to `size` bytes.

        The shape matches `expose_writer_mem_view()`. If less than `size` bytes
        are currently readable, clamp to the amount available rather than raising.
        """
        if self.reader_pos_index is None:
            raise RuntimeError("expose_reader_mem_view called on a writer-only instance")
        
        max_readable = self.get_write_pos() - self.reader_pos
        
        if max_readable > self.buffer_size:
            self.jump_to_writer()
            max_readable = 0
        
        actual_size = min(size, max_readable)

        read_offset = self.int_to_pos(self.reader_pos)

        if read_offset + actual_size <= self.buffer_size: 
            mv1 = self.buffer[read_offset: read_offset + actual_size]
            mv2 = None 
            split = False 

        else: 
            mv1 = self.buffer[read_offset:]
            mv2 = self.buffer[0:actual_size - len(mv1)]
            split = True 

        return (mv1, mv2, actual_size, split)

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """
        Copy bytes from `src` into the exposed writer view(s).

        If `src` is larger than the destination region, copy only the prefix that fits.
        This helper should not publish data by itself; publishing happens when the
        writer position is advanced.
        """
        mv1, mv2, size, split = writer_mem_view
        src_bytes = memoryview(src).cast('B')
        
        first_chunk = min(mv1.nbytes, src_bytes.nbytes)
        mv1[:first_chunk] = src_bytes[:first_chunk] 
        
        if mv2 is not None: 
            second_chunk = src_bytes.nbytes - first_chunk
            mv2[:second_chunk] = src_bytes[first_chunk:first_chunk+ second_chunk]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy bytes from the exposed reader view(s) into `dst`.

        If `dst` is smaller than the readable region, copy only the prefix that fits.
        This helper should not consume data by itself; consumption happens when the
        reader position is advanced.
        """
        mv1, mv2, size, split = reader_mem_view
        dst_bytes = memoryview(dst).cast('B')
        
        first_chunk = min(mv1.nbytes, dst_bytes.nbytes)
        dst_bytes[:first_chunk] = mv1[:first_chunk]

        if mv2 is not None: 
            second_chunk = min(mv2.nbytes, dst_bytes.nbytes - first_chunk)
            dst_bytes[first_chunk:first_chunk+second_chunk] = mv2[:second_chunk]

    def write_array(self, arr: np.ndarray) -> int:
        """
        Write a NumPy array's raw bytes into the shared buffer.

        Return the number of bytes written. If the full array does not fit, the
        contract used by the tests expects this method to return `0`.
        """
        nbytes = arr.nbytes
        writer_mem_view = self.expose_writer_mem_view(nbytes)
        mv1, mv2, actual_size, split = writer_mem_view


        if actual_size < nbytes: 
            return 0 

        self.simple_write(writer_mem_view, arr)
        self.inc_writer_pos(nbytes)
        return nbytes

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        reader_mem_view = self.expose_reader_mem_view(nbytes)
        mv1, mv2, actual_size, split = reader_mem_view
        if actual_size < nbytes:
            return np.empty(0, dtype=dtype)
        if not split:
            arr = np.frombuffer(mv1, dtype=dtype)
        else:
            buf = bytearray(actual_size)
            self.simple_read(reader_mem_view, memoryview(buf))
            arr = np.frombuffer(buf, dtype=dtype)
        self.inc_reader_pos(nbytes)
        return arr

    def _rescan_readers(self) -> None: 
        """
        Pull all reader positions and active flags from shared memory into local cache. 

        When force_rescan = True, this optimization takes place
        """
        header = self.header
        static = self._STATIC
        reader_fields = self._READER_FIELDS
        positions = self._cached_reader_positions
        active = self._cached_reader_active
        slowest = None

        for i in range(self.num_readers):
            slot = static + i * reader_fields
            pos = int(header[slot])
            is_active = bool(header[slot + 1])
            positions[i] = pos
            active[i] = is_active
            if is_active:
                if slowest is None or pos < slowest:
                    slowest = pos
        self._cached_slowest_pos = slowest

