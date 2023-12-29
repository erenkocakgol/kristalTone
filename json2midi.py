import pretty_midi
import json
import os

def json_to_midi(json_directory, output_directory):
    # Belirtilen dizindeki her JSON dosyası için bir döngü oluştur
    for json_file in os.listdir(json_directory):
        if json_file.endswith('.json'):
            # Yeni bir PrettyMIDI nesnesi oluştur
            midi = pretty_midi.PrettyMIDI()

            with open(os.path.join(json_directory, json_file), 'r') as file:
                notes_data = json.load(file)

            # Yeni bir enstrüman oluştur
            instrument = pretty_midi.Instrument(program=0)  # Program numarasını değiştirebilirsiniz

            # JSON'dan not bilgilerini oku ve MIDI'ye dönüştür
            for note_data in notes_data:
                note = pretty_midi.Note(velocity=note_data['velocity'],
                                        pitch=note_data['pitch'],
                                        start=note_data['start'],
                                        end=note_data['end'])
                instrument.notes.append(note)

            # Enstrümanı MIDI'ye ekle
            midi.instruments.append(instrument)

            # MIDI dosyasını kaydet
            midi_filename = os.path.splitext(json_file)[0] + '.mid'
            midi.write(os.path.join(output_directory, midi_filename))

# Örnek kullanım
# json_directory = 'path/to/json_files'
# output_directory = 'path/to/output_midi_files'
# json_to_midi(json_directory, output_directory)
