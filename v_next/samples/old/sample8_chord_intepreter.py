from dataclasses import dataclass
import glob
import os
import re
from typing import Dict, List, Optional

# noinspection PyUnresolvedReferences
from ...src.utils.music import Chord, ChordInterpreter, ChordNotation, ChordWithBass, NoteLanguage


def _format_title(title: str) -> str:
    return re.sub(r"[\s\-_]+", " ", title).lower()


def get_title(file_path: str):
    filename: str = os.path.splitext(os.path.split(file_path)[1])[0]
    m = re.match(r"^[\d\s\-_]*$", filename)
    if m:
        return _format_title(m.group(0))
    m = re.match(r"^(CD\d*[\s\-_]*)?\d*[\s\-_]*(.*?)$", filename)
    if m:
        return _format_title(m.group(2))
    raise ValueError


@dataclass
class FilePairPath:
    wav: str
    lab: str


@dataclass
class ChordRange:
    start: float
    end: float
    chord: ChordWithBass


def main():

    chord_lab_files: Dict = dict()
    for file in glob.glob("." + os.sep + "data" + os.sep + "raw" + os.sep +
                          "**" + os.sep + "chordlab" + os.sep + "**" + os.sep + "*" + os.extsep + "lab",
                          recursive=True):
        chord_lab_files[get_title(file)] = file
    for file in glob.glob("." + os.sep + "data" + os.sep + "raw" + os.sep + "*" + os.extsep + "lab"):
        chord_lab_files[get_title(file)] = file

    for file in sorted(chord_lab_files):
        print(file)

    print()
    lab_and_wav_files: List[FilePairPath] = []
    # for file in glob.glob("." + os.sep + "data" + os.sep + "raw" + os.sep + "*" + os.extsep + "wav"):
    for file in glob.glob("." + os.sep + "data" + os.sep + "raw" + os.sep + "**" + os.sep + "*" + os.extsep + "wav",
                          recursive=True):
        title: str = get_title(file)
        lab_file: Optional[str] = chord_lab_files.get(title)
        if lab_file:
            lab_and_wav_files.append(FilePairPath(file, lab_file))
        else:
            print(title + " is not in dictionary.")

    for file in lab_and_wav_files:
        print()
        print(file)
        # if "Get Back" in file.wav:
        #     print("!")
        with open(file.lab) as f:
            for line in f:
                m = re.match(r"^([\d.]+)\s+([\d.]+)\s+(.*?)$", line)
                if not m:
                    continue
                chord_range: ChordRange = ChordRange(
                    float(m.group(1)), float(m.group(2)),
                    ChordInterpreter.decode_from_str(m.group(3), ChordNotation.MIREX, NoteLanguage.MIDI))
                print("[{:>10.6f} - {:>10.6f}] {:>16s} -> {:<16s} | {}".format(
                    chord_range.start, chord_range.end, m.group(3),
                    ChordInterpreter.encode_to_str(chord_range.chord, ChordNotation.Standard),
                    str(chord_range.chord)))


if __name__ == '__main__':
    main()
