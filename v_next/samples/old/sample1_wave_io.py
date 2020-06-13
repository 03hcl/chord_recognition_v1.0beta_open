# from typing import Iterable, Iterator, Optional, Tuple

# from scipy.io.wavfile import read
from matplotlib import pyplot as plt
# import numpy as np
# import torch

from v_next.src.utils.music import *


def print_each_key_and_signal(s: Signal, shown: int):
    for k in s.keys():
        s_k = s.get(k)
        print(k[1])
        print(s_k[: shown])


def write_wave(s: Signal, filename: str):
    with Writer("./results/{}.wav".format(filename)) as wave_writer:
        wave_writer.write_all(*signal_to_wave(s))


def main():

    print("\u3042")
    print(chr(0x3042))
    print(b"\xe3\x81\x82".decode())
    print("\u266F")
    print(b"\x26\x6F".decode())
    print(chr(0x1d12a))
    print(chr(0x1d12b))
    # print(b"\x01\xd1\x2a".decode("utf-16-be"))

    # note0: Note = NoteInterpreter.to_note("C#", "MIDI")
    # note0b: Note = NoteInterpreter.to_note("C _ ♯", "MIDI")
    # note1: Note = NoteInterpreter.to_note("D#", "mid")
    # note2: Note = NoteInterpreter.to_note("g-sharp", "MIDI")
    # note3: Note = NoteInterpreter.to_note("ド_##", "日本語")
    note0: Note = Note.from_str("C#", "MIDI")
    note0b: Note = Note.from_str("C _ ♯", "MIDI")
    note1: Note = Note.from_str("D#", "mid")
    note2: Note = Note.from_str("g-sharp", "MIDI")
    note3: Note = Note.from_str("ド_##", "日本語")
    print("note0: " + str(note0))
    print("note0b: " + str(note0b))
    print("note1: " + str(note1))
    print("note2: " + str(note2))
    print("note3: " + str(note3))

    pitch1000: Pitch = Pitch.from_frequency(1000)
    print(pitch1000.note_number)
    print(pitch1000.cent)
    print(pitch1000.frequency)
    print(pitch1000.cent_from_note_number(81))

    # conf = Config()
    data_front: int = 32

    # blank: Optional[np.ndarray] = np.array([])
    blank = np.empty(())
    # blank = np.empty(0)
    print("blank is " + ("not " if not blank else "") + "None!")
    # print(len(blank))
    # a = np.append(blank, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    a = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10\x11\x12\x13"
    print(a)
    print(a[: -3])
    print(a[-3:])

    with Reader("./data/raw/20170205.wav") as wave_reader:
        stereo: Stereo = wave_to_signal(*wave_reader.read_all())

    monauralized: Monaural = stereo.monauralize()
    plt.plot(stereo["L"][: data_front])
    plt.plot(stereo["R"][: data_front])
    plt.plot(monauralized["MONO"][: data_front])
    plt.show()

    # normalized_signal: Stereo = stereo.normalize()
    normalized_signal: Stereo = stereo[: data_front].normalize()
    plt.plot(normalized_signal["L"][: data_front])
    plt.plot(normalized_signal["R"][: data_front])
    plt.show()

    print_each_key_and_signal(stereo, data_front)

    print("mono")
    print(monauralized["MONO"][: data_front])

    print("MS:")
    ms_signal: Signal = stereo.get_mid_and_side_channel()
    print_each_key_and_signal(ms_signal, data_front)

    print("Scale (* 100) with Clipping:")
    scaled: Stereo = stereo.scale_with_clipping(100)
    print_each_key_and_signal(scaled, data_front)

    print("Create Various Wave:")
    freq: float = 440
    sampling_rate: int = 48000
    sine: Monaural = Monaural.create_sine_wave(freq, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)
    saw: Monaural = Monaural.create_sawtooth_wave(freq, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)
    tri: Monaural = Monaural.create_triangle_wave(440, phase=np.pi / 2, seconds=5, sampling_rate=sampling_rate)
    print_each_key_and_signal(sine, data_front)
    plt.plot(sine["MONO"][: 400])
    plt.plot(saw["MONO"][: 400])
    plt.plot(tri["MONO"][: 400])
    plt.show()

    print("writing...")

    write_wave(sine, "test_sine_wave")
    write_wave(saw, "test_sawtooth_wave")
    write_wave(tri, "test_triangle_wave")

    write_wave(stereo, "test_triangle_wave")
    write_wave(ms_signal.extract("L-R"), "test_l-r")
    write_wave(stereo.reverse(), "test_reversed")

    print("Finished!")


if __name__ == '__main__':
    main()
