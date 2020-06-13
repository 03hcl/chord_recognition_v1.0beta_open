from dataclasses import dataclass

import torch

from .chord_sequence import ChordSequence


@dataclass
class AnalyzedData:
    data: torch.Tensor
    answer: torch.Tensor
    start_index: int
    end_index: int
    chord_sequence: ChordSequence
    sampling_rate: int
