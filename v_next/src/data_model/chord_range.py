from dataclasses import dataclass

from ..utils.music import ChordWithBass


@dataclass
class ChordRange:
    start: int
    end: int
    chord: ChordWithBass
    raw_str: str
