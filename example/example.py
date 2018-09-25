import sys, os # You don't need these libraries for anything other than importing the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory so that we can import MIDIconverter
#                                                                               This step usually isn't necessary

from MIDIconverter import MConv

# Plays Simple and Clean by Hikaru Utada
# First read the midi, then write to text
# Then read the text file, and write to wave (and midi)
# Then play it

a = MConv(volume=0.5, sample_rate=44100) # Creates a new Midi converter
#                       (The first argument represents the amplitude multiplier DEFAULT: 0.5)
#                       (The second argument represents the sampling rate for writing to wave format / playing audio DEFAULT: 44100 Hz)

a.text_file_accuracy = 64 # Changes the accuracy of reading and writing to text files, higher means more accurate (Default is 32)

# Change the note envelope (Changes how each note sounds) (Default values shown)
a.A = (0.001, 1)
a.D = (0.5, 0.5)
a.S = (0.75, 0.25)

a.read_midi('Clean.mid', wav=False) # Reads a midi and saves data to write to wav format (Writing to .wav can take a long time so disable this if you don't need it)
a.write_text('Clean.txt') # Writes a text file that an AI can train on

""" All possible write/play functions listed below """
# a.write_wave('Clean.wav') # Writes a wave file
# a.write_matlab('Clean.mat') # Writes a .mat file which can be read by matlab
# a.play() # Plays the audio
# a.write_midi('Clean.mid') # Writes a MIDI file

a.terminate() # Terminate the converter




a = MConv() # Do not read twice using the same MConv object, create a new one and read again
a.text_file_accuracy = 64 # Must match the write accuracy

a.read_text('Clean.txt', midi=True, wav=True)
a.write_midi('Clean_2.mid')
a.write_wave('Clean_2.wav')

a.play()

a.terminate() # Terminate the converter
