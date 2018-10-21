import pyaudio
import numpy as np
import time
from midiutil import MIDIFile
import wavio
import mido
import glob
import os
import scipy.io as sio
from numpy.core.records import fromarrays

def create_matlab_array(struct_array):
    myrec = []
    for s in struct_array:
        values = list(s.values())
        myrec.append(values)

    keys = list(s.keys())
    myrec = np.array(myrec, dtype=object)
    myrec = myrec.T
    return fromarrays(myrec, names = keys)

def sin(x):
    return np.sin(x)

def square(x):
    return np.clip(np.sin(x) * 1000, -1, 1) * 100

class MConv:
    text_file_accuracy = 32
    A = (0.001, 1)
    D = (0.5, 0.5)
    S = (0.75, 0.25)

    def __init__(self, volume=0.5, sample_rate=44100):        
        self.volume = volume
        self.sample_rate = sample_rate
        self.init()

        self.p = pyaudio.PyAudio()
        self.stream  = self.p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.sample_rate,
                        output=True)


        # This class will eventually allow the user to specify their own notes, so here is a dictionary of all possible notes to there frequencies
        self.notes_freq = {'G9': 12543.85, 'F#9/Gb9': 11839.82, 'F9': 11175.30, 'E9': 10548.08, 'D#9/Eb9': 9956.06, 'D9': 9397.27, 'C#9/Db9': 8869.84, 'C9': 8372.02, 'B8': 7902.13, 'A#8/Bb8': 7458.62, 'A8': 7040.00, 'G#8/Ab8': 6644.88, 'G8': 6271.93, 'F#8/Gb8': 5919.91, 'F8': 5587.65, 'E8': 5274.04, 'D#8/Eb8': 4978.03, 'D8': 4698.64, 'C#8/Db8': 4434.92, 'C8': 4186.01, 'B7': 3951.07, 'A#7/Bb7': 3729.31, 'A7': 3520.00, 'G#7/Ab7': 3322.44, 'G7': 3135.96, 'F#7/Gb7': 2959.96, 'F7': 2793.83, 'E7': 2637.02, 'D#7/Eb7': 2489.02, 'D7': 2349.32, 'C#7/Db7': 2217.46, 'C7': 2093.00, 'B6': 1975.53, 'A#6/Bb6': 1864.66, 'A6': 1760.00, 'G#6/Ab6': 1661.22, 'G6': 1567.98, 'F#6/Gb6': 1479.98, 'F6': 1396.91, 'E6': 1318.51, 'D#6/Eb6': 1244.51, 'D6': 1174.66, 'C#6/Db6': 1108.73, 'C6': 1046.50, 'B5': 987.77, 'A#5/Bb5': 932.33, 'A5': 880.00, 'G#5/Ab5': 830.61, 'G5': 783.99, 'F#5/Gb5': 739.99, 'F5': 698.46, 'E5': 659.26, 'D#5/Eb5': 622.25, 'D5': 587.33, 'C#5/Db5': 554.37, 'C5': 523.25, 'B4': 493.88, 'A#4/Bb4': 466.16, 'A4': 440.00, 'G#4/Ab4': 415.30, 'G4': 392.00, 'F#4/Gb4': 369.99, 'F4': 349.23, 'E4': 329.63, 'D#4/Eb4': 311.13, 'D4': 293.66, 'C#4/Db4': 277.18, 'C4': 261.63, 'B3': 246.94, 'A#3/Bb3': 233.08, 'A3': 220.00, 'G#3/Ab3': 207.65, 'G3': 196.00, 'F#3/Gb3': 185.00, 'F3': 174.61, 'E3': 164.81, 'D#3/Eb3': 155.56, 'D3': 146.83, 'C#3/Db3': 138.59, 'C3': 130.81, 'B2': 123.47, 'A#2/Bb2': 116.54, 'A2': 110.00, 'G#2/Ab2': 103.83, 'G2': 98.00, 'F#2/Gb2': 92.50, 'F2': 87.31, 'E2': 82.41, 'D#2/Eb2': 77.78, 'D2': 73.42, 'C#2/Db2': 69.30, 'C2': 65.41, 'B1': 61.74, 'A#1/Bb1': 58.27, 'A1': 55.00, 'G#1/Ab1': 51.91, 'G1': 49.00, 'F#1/Gb1': 46.25, 'F1': 43.65, 'E1': 41.20, 'D#1/Eb1': 38.89, 'D1': 36.71, 'C#1/Db1': 34.65, 'C1': 32.70, 'B0': 30.87, 'A#0/Bb0': 29.14, 'A0': 27.50, '': 0.0 }

        # A dictionary relating notes to MIDI note numbers
        self.midi_notes = {'G9': 127, 'F#9/Gb9': 126, 'F9': 125, 'E9': 124, 'D#9/Eb9': 123, 'D9': 122, 'C#9/Db9': 121, 'C9': 120, 'B8': 119, 'A#8/Bb8': 118, 'A8': 117, 'G#8/Ab8': 116, 'G8': 115, 'F#8/Gb8': 114, 'F8': 113, 'E8': 112, 'D#8/Eb8': 111, 'D8': 110, 'C#8/Db8': 109, 'C8': 108, 'B7': 107, 'A#7/Bb7': 106, 'A7': 105, 'G#7/Ab7': 104, 'G7': 103, 'F#7/Gb7': 102, 'F7': 101, 'E7': 100, 'D#7/Eb7': 99, 'D7': 98, 'C#7/Db7': 97, 'C7': 96, 'B6': 95, 'A#6/Bb6': 94, 'A6': 93, 'G#6/Ab6': 92, 'G6': 91, 'F#6/Gb6': 90, 'F6': 89, 'E6': 88, 'D#6/Eb6': 87, 'D6': 86, 'C#6/Db6': 85, 'C6': 84, 'B5': 83, 'A#5/Bb5': 82, 'A5': 81, 'G#5/Ab5': 80, 'G5': 79, 'F#5/Gb5': 78, 'F5': 77, 'E5': 76, 'D#5/Eb5': 75, 'D5': 74, 'C#5/Db5': 73, 'C5': 72, 'B4': 71, 'A#4/Bb4': 70, 'A4': 69, 'G#4/Ab4': 68, 'G4': 67, 'F#4/Gb4': 66, 'F4': 65, 'E4': 64, 'D#4/Eb4': 63, 'D4': 62, 'C#4/Db4': 61, 'C4': 60, 'B3': 59, 'A#3/Bb3': 58, 'A3': 57, 'G#3/Ab3': 56, 'G3': 55, 'F#3/Gb3': 54, 'F3': 53, 'E3': 52, 'D#3/Eb3': 51, 'D3': 50, 'C#3/Db3': 49, 'C3': 48, 'B2': 47, 'A#2/Bb2': 46, 'A2': 45, 'G#2/Ab2': 44, 'G2': 43, 'F#2/Gb2': 42, 'F2': 41, 'E2': 40, 'D#2/Eb2': 39, 'D2': 38, 'C#2/Db2': 37, 'C2': 36, 'B1': 35, 'A#1/Bb1': 34, 'A1': 33, 'G#1/Ab1': 32, 'G1': 31, 'F#1/Gb1': 30, 'F1': 29, 'E1': 28, 'D#1/Eb1': 27, 'D1': 26, 'C#1/Db1': 25, 'C1': 24, 'B0': 23, 'A#0/Bb0': 22, 'A0': 21 }

        # A dictionary relating MIDI note numbers to notes
        self.from_midi = {127: 'G9', 126: 'F#9/Gb9', 125: 'F9', 124: 'E9', 123: 'D#9/Eb9', 122: 'D9', 121: 'C#9/Db9', 120: 'C9', 119: 'B8', 118: 'A#8/Bb8', 117: 'A8', 116: 'G#8/Ab8', 115: 'G8', 114: 'F#8/Gb8', 113: 'F8', 112: 'E8', 111: 'D#8/Eb8', 110: 'D8', 109: 'C#8/Db8', 108: 'C8', 107: 'B7', 106: 'A#7/Bb7', 105: 'A7', 104: 'G#7/Ab7', 103: 'G7', 102: 'F#7/Gb7', 101: 'F7', 100: 'E7', 99: 'D#7/Eb7', 98: 'D7', 97: 'C#7/Db7', 96: 'C7', 95: 'B6', 94: 'A#6/Bb6', 93: 'A6', 92: 'G#6/Ab6', 91: 'G6', 90: 'F#6/Gb6', 89: 'F6', 88: 'E6', 87: 'D#6/Eb6', 86: 'D6', 85: 'C#6/Db6', 84: 'C6', 83: 'B5', 82: 'A#5/Bb5', 81: 'A5', 80: 'G#5/Ab5', 79: 'G5', 78: 'F#5/Gb5', 77: 'F5', 76: 'E5', 75: 'D#5/Eb5', 74: 'D5', 73: 'C#5/Db5', 72: 'C5', 71: 'B4', 70: 'A#4/Bb4', 69: 'A4', 68: 'G#4/Ab4', 67: 'G4', 66: 'F#4/Gb4', 65: 'F4', 64: 'E4', 63: 'D#4/Eb4', 62: 'D4', 61: 'C#4/Db4', 60: 'C4', 59: 'B3', 58: 'A#3/Bb3', 57: 'A3', 56: 'G#3/Ab3', 55: 'G3', 54: 'F#3/Gb3', 53: 'F3', 52: 'E3', 51: 'D#3/Eb3', 50: 'D3', 49: 'C#3/Db3', 48: 'C3', 47: 'B2', 46: 'A#2/Bb2', 45: 'A2', 44: 'G#2/Ab2', 43: 'G2', 42: 'F#2/Gb2', 41: 'F2', 40: 'E2', 39: 'D#2/Eb2', 38: 'D2', 37: 'C#2/Db2', 36: 'C2', 35: 'B1', 34: 'A#1/Bb1', 33: 'A1', 32: 'G#1/Ab1', 31: 'G1', 30: 'F#1/Gb1', 29: 'F1', 28: 'E1', 27: 'D#1/Eb1', 26: 'D1', 25: 'C#1/Db1', 24: 'C1', 23: 'B0', 22: 'A#0/Bb0', 21: 'A0', }

        # A dictionary relating MIDI note numbers to frequencies
        self.midi_freq = {127: 12543.85, 126: 11839.82, 125: 11175.30, 124: 10548.08, 123: 9956.06, 122: 9397.27, 121: 8869.84, 120: 8372.02, 119: 7902.13, 118: 7458.62, 117: 7040.00, 116: 6644.88, 115: 6271.93, 114: 5919.91, 113: 5587.65, 112: 5274.04, 111: 4978.03, 110: 4698.64, 109: 4434.92, 108: 4186.01, 107: 3951.07, 106: 3729.31, 105: 3520.00, 104: 3322.44, 103: 3135.96, 102: 2959.96, 101: 2793.83, 100: 2637.02, 99: 2489.02, 98: 2349.32, 97: 2217.46, 96: 2093.00, 95: 1975.53, 94: 1864.66, 93: 1760.00, 92: 1661.22, 91: 1567.98, 90: 1479.98, 89: 1396.91, 88: 1318.51, 87: 1244.51, 86: 1174.66, 85: 1108.73, 84: 1046.50, 83: 987.77, 82: 932.33, 81: 880.00, 80: 830.61, 79: 783.99, 78: 739.99, 77: 698.46, 76: 659.26, 75: 622.25, 74: 587.33, 73: 554.37, 72: 523.25, 71: 493.88, 70: 466.16, 69: 440.00, 68: 415.30, 67: 392.00, 66: 369.99, 65: 349.23, 64: 329.63, 63: 311.13, 62: 293.66, 61: 277.18, 60: 261.63, 59: 246.94, 58: 233.08, 57: 220.00, 56: 207.65, 55: 196.00, 54: 185.00, 53: 174.61, 52: 164.81, 51: 155.56, 50: 146.83, 49: 138.59, 48: 130.81, 47: 123.47, 46: 116.54, 45: 110.00, 44: 103.83, 43: 98.00, 42: 92.50, 41: 87.31, 40: 82.41, 39: 77.78, 38: 73.42, 37: 69.30, 36: 65.41, 35: 61.74, 34: 58.27, 33: 55.00, 32: 51.91, 31: 49.00, 30: 46.25, 29: 43.65, 28: 41.20, 27: 38.89, 26: 36.71, 25: 34.65, 24: 32.70, 23: 30.87, 22: 29.14, 21: 27.50, 20: 25.96, 19: 24.50, 18: 23.12, 17: 21.83, 16: 20.60, 15: 19.45, 14: 18.35, 13: 17.32, 12: 16.35, 11: 15.43, 10: 14.57, 9: 13.75, 8: 12.98, 7: 12.25, 6: 11.56, 5: 10.91, 4: 10.30, 3: 9.72, 2: 9.18, 1: 8.66, 0: 8.18}

    # Creates an ADSR profile for the notes, A, D, and S can be set by the user to achieve different effects
    def ADSR(self, pos):
        prev = (0, 0)
        for curr in [self.A, self.D, self.S, (1, 0)]:
            if curr[0] >= pos:
                delta = pos - prev[0]
                distance = curr[0] - prev[0]

                ratio = delta / distance

                amplitude = prev[1] * (1 - ratio) + curr[1] * ratio
                return amplitude
            prev = curr

    # Generates the waveform for a single note
    def generate_wave(self, note, start_time, dur_time, midi=True, wav=True, func=square):
        if start_time < 0:
            start_time = self.time - self.key_time

        if type(note) == int:
            freq = self.midi_freq[note]
        else:
            freq = self.notes_freq[note]

        duration = dur_time * self.sample_rate
        start_sample = round((self.key_time + start_time) * self.sample_rate)
        end_time = self.key_time + start_time + duration

        wave = func(2*np.pi*np.arange(duration)*freq/self.sample_rate)
        duration = len(wave)
        end_sample = start_sample + duration


        if duration > 0 and midi:
            if type(note) == int:
                self.midi.addNote(0, 0, note, self.key_time + start_time, dur_time, int(self.volume * 127))
            elif note != '':
                self.midi.addNote(0, 0, self.midi_notes[note], self.key_time + start_time, dur_time, int(self.volume * 127))

        self.key_time += start_time
        if end_time > self.time:
            self.time = end_time

        if duration > 0.0 and wav:
            for i in range(duration):
                wave[i] *= self.ADSR(i/duration)

            return (wave.astype(np.float32), start_sample, end_sample)
        else:
            return None

    # Generates the waveform for a series of notes
    def generate_wave_stream(self, notes, midi=True, wav=True):
        self.time = 0
        self.key_time = 0

        wave = []
        for n in notes:
            if len(n) == 3:
                w = self.generate_wave(n[0], n[1], n[2], midi=midi, wav=wav)
            else:
                w = self.generate_wave(n[0], -1, n[1], midi=midi, wav=wav)

            if w is not None:
                wave.append(w)

        max_end = 0
        for w in wave:
            _, _, end = w
            if end > max_end:
                max_end = end

        self.samples = np.zeros(max_end, dtype=np.float32)

        for w in wave:
            samples, start, end = w
            self.samples[start:end] += samples

        self.samples /= max(self.samples)

    # Plays the notes using PyAudio
    def play(self):
        #self.generate_wave_stream(self.notes)

        samples = self.samples * self.volume

        i = 0
        for i in range(0, len(samples), 1024):
            self.stream.write(samples[i: i + 1024].tobytes())

        self.stream.write(samples[i:].tobytes())

    def read_error(self):
        print("MConv can only read from one file, to read another, please create a new MConv object to read it")

    # Reads from MIDI formats
    def read_midi(self, file_name, wav=True):
        self.read_reset()

        mid = mido.MidiFile(file_name)

        notes = []
        last_on = 0
        time = 0
        midi_list = list(mid)

        first = True

        for i, msg in enumerate(midi_list):
            time += msg.time
            if msg.type == 'note_on' and msg.velocity != 0:
                if first:
                    time = 0.0
                    first = False

                delta_from_last = time - last_on

                sub_time = time
                duration = -1

                for j, sub_msg in enumerate(midi_list[i+1:]):
                    sub_time += sub_msg.time

                    if (sub_msg.type == 'note_on' or sub_msg.type == 'note_off') and sub_msg.note == msg.note:
                        duration = sub_time - time
                        break

                notes.append((msg.note, delta_from_last, duration))

                last_on = time

        self.notes = notes

        if wav:
            self.generate_wave_stream(self.notes, midi=False, wav=True)

    # Reads from text format
    def read_text(self, file_name, midi=True, wav=True):
        self.read_reset()

        with open(file_name, 'r') as f:
            content = f.read()

            self.notes = []
            j = 0

            for i in range(0, len(content)):
                try:
                    start, midi_char, midi_delta_time, midi_duration = content[i: i + 4]
                except ValueError as e:
                    break

                if start != chr(0):
                    continue
                else:
                    j += 1

                midi_char = ord(midi_char) - 1
                midi_char = max(0, midi_char)

                midi_delta_time = (ord(midi_delta_time) - 1) / self.text_file_accuracy
                midi_delta_time = max(0, midi_delta_time)
                midi_duration = (ord(midi_duration) - 1) / self.text_file_accuracy
                midi_duration = max(0, midi_duration)

                self.notes.append((midi_char, midi_delta_time, midi_duration))

            if midi or wav:
                self.generate_wave_stream(self.notes, midi=midi, wav=wav)

    # Write to MIDI format
    def write_midi(self, file_name):
        with open(file_name, "wb") as output_file:
            self.midi.writeFile(output_file)

    # Write to .mat format
    def write_matlab(self, file_name):
        time = 0
        mat_notes = []
        for note in self.notes:
            time += note[1] # Update the time

            struct = {
                'duration': note[2],
                'start_time': time,
                'note_number': note[0],
            }

            mat_notes.append(struct)

        mat_array = create_matlab_array(mat_notes)
        sio.savemat(file_name, {file_name[:-4]:mat_array})

    # Write to text format
    def write_text(self, file_name, append=False):
        content = ''
        if append and os.path.exists(file_name):
            with open(file_name, 'r') as f:
                content = f.read()

        with open(file_name, 'w') as f:
            f.write(content)

            for note in self.notes:
                midi_char = chr(note[0] + 1)
                midi_delta_time = chr(round(note[1] * self.text_file_accuracy) + 1)
                midi_duration = chr(round(note[2] * self.text_file_accuracy) + 1)

                #print(note[1], ord(midi_delta_time))
                f.write(chr(0) + midi_char + midi_delta_time + midi_duration)

    # Write to .wav format
    def write_wav(self, file_name):
        samples = self.samples * self.volume
        wavio.write(file_name, samples, self.sample_rate, sampwidth=3)

    # Called whenever we read, needed in order to reset the variables so we can read again
    def read_reset(self):
        self.init()

    # Initializes variables that might need to be re-initialized for resets
    def init(self):
        self.time = 0
        self.key_time = 0 # Marker for the start of the last note, all note starts will be relative to this
        self.notes = []

        # Prepare to write as a midi
        self.midi = MIDIFile(deinterleave=False)
        self.midi.addTempo(0, 0, 60)

        self.samples = np.zeros(100)

    # End the stream and end player
    def terminate(self):
        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()
