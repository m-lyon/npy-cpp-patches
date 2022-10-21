'''NumPy Patcher'''
from typing import List, Tuple

from numpy import double, float32, int32, int64

class PatcherDouble:
    def __init__(self) -> None: ...
    def get_patch(
        self,
        fpath: str,
        qidx: Tuple[int, ...],
        pshape: Tuple[int, ...],
        pstride: Tuple[int, ...],
        pnum: int,
    ) -> List[double]: ...

class PatcherFloat:
    def __init__(self) -> None: ...
    def get_patch(
        self,
        fpath: str,
        qidx: Tuple[int, ...],
        pshape: Tuple[int, ...],
        pstride: Tuple[int, ...],
        pnum: int,
    ) -> List[float32]: ...

class PatcherInt:
    def __init__(self) -> None: ...
    def get_patch(
        self,
        fpath: str,
        qidx: Tuple[int, ...],
        pshape: Tuple[int, ...],
        pstride: Tuple[int, ...],
        pnum: int,
    ) -> List[int32]: ...

class PatcherLong:
    def __init__(self) -> None: ...
    def get_patch(
        self,
        fpath: str,
        qidx: Tuple[int, ...],
        pshape: Tuple[int, ...],
        pstride: Tuple[int, ...],
        pnum: int,
    ) -> List[int64]: ...
