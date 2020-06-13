from ...src.utils.music import *


def main():

    p_float: Pitch = Pitch.from_frequency(256)
    print(p_float.show_note_details())
    print(p_float.show_note_details(is_shortened=False))

    print(Pitch.from_frequency(440.001).show_note_details())
    print(Pitch.from_frequency(440.001).show_note_details(is_shortened=False))
    print(Pitch(69, cent=0.99).show_note_details())
    print(Pitch(69, cent=0.99).show_note_details(is_shortened=False))

    print(Pitch.from_note_and_octave(Note.B, 3).show_note_details())
    print(Pitch.from_note_and_octave(Note.C, 4).show_note_details())
    print(Pitch.from_note_and_octave(Note.C_SHARP, 4).show_note_details())
    print(Pitch.from_note_and_octave(Note.G_SHARP, 4).show_note_details())
    print(Pitch.from_note_and_octave(Note.A, 4).show_note_details())
    print(Pitch.from_note_and_octave(Note.A_SHARP, 4).show_note_details())

    stride: int = 2000
    freq: float = 440
    sampling_rate: int = 48000
    standard_pitch: float = 440

    tri: Monaural = Monaural.create_triangle_wave(freq, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)
    cqt: VQT = tri.cqt(stride=stride)

    ps1 = Pitch.from_frequency((500, 1000), standard_pitch=standard_pitch)
    print(len(ps1))
    ps2 = Pitch.from_frequency({500, 1000}, standard_pitch=standard_pitch)
    print(len(ps2))
    # noinspection PyTypeChecker
    ps3 = Pitch.from_frequency(cqt.frequencies.tolist(), standard_pitch=standard_pitch)
    print(len(ps3))

    cqt_notes: np.ndarray = Pitch.from_frequency(cqt.frequencies, standard_pitch=standard_pitch)
    for cqt_note in cqt_notes:
        print(str(cqt_note))


if __name__ == '__main__':
    main()
