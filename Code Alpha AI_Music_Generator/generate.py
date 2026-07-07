import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import instrument, note, chord, stream

# Load trained model
model = load_model("model.keras")

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

sequence_length = 100

# Choose a random starting point
start = np.random.randint(0, len(notes) - sequence_length)

pattern = [note_to_int[n] for n in notes[start:start + sequence_length]]

prediction_output = []

print("Generating music...")

# Generate 200 notes
for _ in range(200):

    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(pitchnames))

    prediction = model.predict(prediction_input, verbose=0)

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:]

offset = 0
output_notes = []

# Convert notes back to MIDI
for pattern in prediction_output:

    if "." in pattern or pattern.isdigit():

        notes_in_chord = pattern.split(".")
        chord_notes = []

        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)

    else:

        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write("midi", fp="output.mid")

print("Music generated successfully!")
print("Saved as output.mid")