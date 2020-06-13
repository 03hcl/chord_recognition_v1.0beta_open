from collections.abc import Sequence
from typing import Iterable, List, Union

from .chord_range import ChordRange


class ChordSequence(Sequence):

    # region Property

    @property
    def start_index(self) -> int:
        return self._raw[0].start

    @property
    def end_index(self) -> int:
        return self._raw[-1].end

    # endregion

    def __init__(self, value: Iterable[ChordRange]):
        self._raw: List[ChordRange] = sorted(value, key=lambda v: v.start)

    def __getitem__(self, i: Union[int, slice]) -> Union[ChordRange, List[ChordRange]]:
        if issubclass(type(i), int):
            return self._raw[self._search_raw_list_index(i)]
        elif issubclass(type(i), slice):
            result: list = []
            s: slice = slice(*i.indices(self.end_index - self.start_index))
            s = slice(s.start + self.start_index, s.stop + self.start_index, s.step)
            if s.step > 0:
                i: int = s.start
                i_raw: int = self._search_raw_list_index(i)
                while True:
                    cr: ChordRange = self._raw[i_raw]
                    length: int = (min(s.stop, cr.end) - i) // s.step
                    result.extend([cr] * length)
                    if s.stop <= cr.end:
                        return result
                    i += length * s.step
                    i_raw += 1
            elif s.step < 0:
                i: int = s.stop
                i_raw: int = self._search_raw_list_index(i)
                while True:
                    cr: ChordRange = self._raw[i_raw]
                    length: int = (i - max(s.start, cr.start)) // s.step
                    result.extend([cr] * length)
                    if s.start >= cr.start:
                        return result
                    i -= length * s.step
                    i_raw -= 1

    def __len__(self) -> int:
        return self.end_index - self.start_index

    def _search_raw_list_index(self, i: int):
        if i < self.start_index or i >= self.end_index:
            raise IndexError
        l: int = 0
        r: int = len(self._raw) - 1
        while True:
            value: int = l + (r - l) // 2
            # noinspection PyPep8
            if self._raw[value].start <= i < self._raw[value].end:
                return value
            elif i < self._raw[value].start:
                r = value - 1
            elif i >= self._raw[value].end:
                l = value + 1
            elif l == r:
                raise IndexError
