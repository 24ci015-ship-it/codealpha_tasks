from music21 import converter, instrument, note, chord
import glob
import pickle

# List to store all extracted notes
notes = []

# Read all MIDI files from the dataset folder
for file in glob.glob("dataset/*.mid"):
    print("Processing:", file)

    try:
        midi = converter.parse(file)

        # Separate instruments if available
        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        # Extract notes and chords
        for element in notes_to_parse:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Save notes into a pickle file
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("\n--------------------------------")
print("Preprocessing Completed!")
print("Total Notes Extracted:", len(notes))
print("Saved as notes.pkl")
print("--------------------------------") 