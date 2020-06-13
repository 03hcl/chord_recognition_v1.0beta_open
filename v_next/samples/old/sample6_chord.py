# noinspection PyUnresolvedReferences
from ...src.utils.music import Chord, ChordInterpreter, NoneChord, Note, Interval


def main():

    chord_c_maj: Chord = Chord(Note.C, Note.E, Note.G)
    print(chord_c_maj.interval_3rd)
    print(chord_c_maj.interval_5th)

    chord_gb_min: Chord = Chord("Gb", "B", "Db")
    print("3rd: {}".format(chord_gb_min.interval_3rd))
    print("5th: {}".format(chord_gb_min.interval_5th))
    print("7th: {}".format(chord_gb_min.interval_7th))
    print("9th: {}".format(chord_gb_min.interval_9th))

    chord_c_add9: Chord = Chord("C", "E", "G", ninth="D")
    print(chord_c_add9.interval_9th)
    print("length: {}".format(len(chord_c_add9)))
    print("in D: {}".format(Note.D in chord_c_add9))
    print("in D#: {}".format(Note.D_SHARP in chord_c_add9))
    print("7th: {}".format(chord_c_add9[7]))
    print("9th: {}".format(chord_c_add9[9]))
    print(len(chord_c_add9))

    for n in chord_c_add9:
        # print(type(n))
        # num, note = n
        # print(num)
        # print(note)
        print(n)

    chord_c_add9_2: Chord = Chord("C", Interval.M3, Interval.P5, None, Interval.M9)
    print(chord_c_add9 == chord_c_add9_2)

    chord_test1_str: str = "Dm7(9,11)"
    chord_test1: str = Chord.from_str(chord_test1_str)

    print(chord_test1)

    chord_c_omit_root: Chord = Chord(Note.C, Interval.M3, Interval.P5, omits_root=True)
    print(chord_c_omit_root)

    none_chord: Chord = NoneChord
    none_chord_d: Chord = Chord(Note.D, None, None, omits_root=True)
    print(none_chord)
    print(none_chord == none_chord_d)


if __name__ == '__main__':
    main()
