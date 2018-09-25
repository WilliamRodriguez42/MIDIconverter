# MIDIconverter
Simple class to convert MIDI to text (and other formats) for training character level neural networks.

This MConv class can convert MIDI files to text, wav, and matlab (.mat) formats, and can play them using PyAudio. It can also read the text files that it generates and convert them back into MIDI format or directly to wav.

Goals:
* *Optimize the sample generation functions (writing to wav is by far the slowest part)*
* *Multithread the sample addition process to make it faster*
* *Find a better way to represent times in text files to reduce large vocab sizes in neural networks*
* *Automate the process of converting and appending all MIDI's in a folder to a text file*
