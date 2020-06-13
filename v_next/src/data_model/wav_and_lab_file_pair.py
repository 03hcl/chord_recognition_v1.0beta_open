from dataclasses import dataclass


@dataclass
class WavAndLabFilePair:
    key: str
    wav: str
    lab: str
    gap: float
