import pretty_midi
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

def midi_to_json(midi_file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        midi_file_directory = os.path.dirname(midi_file_path)

        for i, instrument in enumerate(midi_data.instruments):
            notes_data = []
            for note in instrument.notes:
                note_data = {
                    'start': note.start,
                    'end': note.end,
                    'pitch': note.pitch,
                    'velocity': note.velocity
                }
                notes_data.append(note_data)

            instrument_json_file_path = f'{midi_file_path.replace(".mid", "")}_instrument_{i}.json'
            with open(instrument_json_file_path, 'w') as json_file:
                json.dump(notes_data, json_file, indent=4)
            
            logging.info(f"Data for instrument {i} saved to {instrument_json_file_path}")

    except Exception as e:
        logging.error(f"Error in MIDI to JSON conversion: {e}")

# Usage example (commented for direct script execution)
# midi_to_json('path/to/zelda.mid')
