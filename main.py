import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging
import os
import numpy as np
import pretty_midi

from midi2wav import combine_midi_to_wav
from preprocessing import MIDIDataset, to_tensor
from midi2json import midi_to_json
from json2midi import json_to_midi
from models import MIDIModel, EarlyStopping
from parameters import Hiperparameters

def predict_json_save_json(model, test_midi_path, predicted_track_path, json_index, batch_size):
    predicted_notes = []
    model.eval()
    with torch.no_grad():
        test_dataset = MIDIDataset(json_folder_path=test_midi_path, transform=to_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        for batch in test_loader:
            inputs = batch
            outputs = model(inputs)
            for output in outputs:
                predicted_note = {
                    'start': output[0].item(),
                    'end': output[1].item(),
                    'pitch': round(output[2].item()),
                    'velocity': round(output[3].item())
                }
                predicted_notes.append(predicted_note)
    
    with open(predicted_track_path + f"/instrument_{json_index}.json", 'w') as json_file:
        json.dump(predicted_notes, json_file, indent=4)


def quantize_note_start_end(note, quantize_to=1/32):
    # Notanın başlangıç ve bitiş zamanlarını belirli bir aralığa yuvarla
    note.start = round(note.start / quantize_to) * quantize_to
    note.end = round(note.end / quantize_to) * quantize_to
    return note

def quantize_midi_file(input_directory, output_directory, quantize_to):
    # Belirtilen dizindeki her MIDI dosyası için bir döngü oluştur
    for file in os.listdir(input_directory):
        if file.endswith('.mid') or file.endswith('.midi'):
            input_file_path = os.path.join(input_directory, file)
            output_file_path = os.path.join(output_directory, file)

            # MIDI dosyasını yükle
            midi_data = pretty_midi.PrettyMIDI(input_file_path)

            # Her notayı kuantize et
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    note = quantize_note_start_end(note, quantize_to)

            # Kuantize edilmiş MIDI dosyasını kaydet
            midi_data.write(output_file_path)

def midi_number_to_note_name(midi_number):
    # MIDI notaları ve isimleri arasındaki eşleşme
    midi_note_mapping = {
        21: 'A0', 22: 'A#0', 23: 'B0',
        24: 'C1', 25: 'C#1', 26: 'D1', 27: 'D#1', 28: 'E1', 29: 'F1', 30: 'F#1', 31: 'G1', 32: 'G#1',
        33: 'A1', 34: 'A#1', 35: 'B1',
        36: 'C2', 37: 'C#2', 38: 'D2', 39: 'D#2', 40: 'E2', 41: 'F2', 42: 'F#2', 43: 'G2', 44: 'G#2',
        45: 'A2', 46: 'A#2', 47: 'B2',
        48: 'C3', 49: 'C#3', 50: 'D3', 51: 'D#3', 52: 'E3', 53: 'F3', 54: 'F#3', 55: 'G3', 56: 'G#3',
        57: 'A3', 58: 'A#3', 59: 'B3',
        60: 'C4', 61: 'C#4', 62: 'D4', 63: 'D#4', 64: 'E4', 65: 'F4', 66: 'F#4', 67: 'G4', 68: 'G#4',
        69: 'A4', 70: 'A#4', 71: 'B4',
        72: 'C5', 73: 'C#5', 74: 'D5', 75: 'D#5', 76: 'E5', 77: 'F5', 78: 'F#5', 79: 'G5', 80: 'G#5',
        81: 'A5', 82: 'A#5', 83: 'B5',
        84: 'C6', 85: 'C#6', 86: 'D6', 87: 'D#6', 88: 'E6', 89: 'F6', 90: 'F#6', 91: 'G6', 92: 'G#6',
        93: 'A6', 94: 'A#6', 95: 'B6',
        96: 'C7', 97: 'C#7', 98: 'D7', 99: 'D#7', 100: 'E7', 101: 'F7', 102: 'F#7', 103: 'G7', 104: 'G#7',
        105: 'A7', 106: 'A#7', 107: 'B7', 108: 'C8'
    }

    return midi_note_mapping.get(midi_number, None)

def train_model(model, dataset_path, epochs, batch_size, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=25, verbose=True)

    # MIDI dosyalarını JSON'a dönüştür ve Dataset oluştur
    #midi_files = [f for f in os.listdir(dataset_path) if f.endswith('.mid')]

    #for midi_file in midi_files:
        #midi_to_json(os.path.join(dataset_path, midi_file))
    
    train_dataset = MIDIDataset(json_folder_path=dataset_path, transform=to_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = MIDIDataset(json_folder_path=dataset_path, transform=to_tensor)
    val_dataset.use_validation()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch, batch
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validasyon kaybını hesapla
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch, batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss}')
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # En iyi modeli yükle
    model.load_state_dict(torch.load('model.bin'))
    
# Örnek eğitim döngüsü kullanımı
# train_model(model, train_loader, val_loader, epochs=10)

def main():
    try:
        # MIDI dosyalarının bulunduğu yol

        model = MIDIModel()
        print("TRAIN STARTING...")
        
        try:
            print("Varolan model yükleniyor...")
            model.load_state_dict(torch.load('model.bin'))
        except:
            print("Model yeniden yaratılacak...")
            pass

        train_model(model, Hiperparameters.dataset_path, epochs=Hiperparameters.epochs, batch_size=Hiperparameters.batch_size, learning_rate=Hiperparameters.learning_rate)
        
        model.load_state_dict(torch.load('model.bin'))

        # MIDI dosyalarını JSON'a dönüştür ve Dataset oluştur
        midi_files = [f for f in os.listdir(Hiperparameters.test_midi_path) if f.endswith('.mid')]

        for midi_file in midi_files:
            midi_to_json(os.path.join(Hiperparameters.test_midi_path, midi_file))

        # Modelin tahminlerini kaydet
        predicted_notes_midi = 'predicted'
        print("PREDICT SAVING...")
        json_index = 0
        for n in range(len([jsonko for jsonko in os.listdir(Hiperparameters.test_midi_path) if jsonko.endswith('.json')])):
            predict_json_save_json(model=model, test_midi_path=os.getcwd() + "/" + Hiperparameters.test_midi_path, predicted_track_path=Hiperparameters.predicted_track_path, json_index=json_index, batch_size=Hiperparameters.batch_size)
            json_index = json_index + 1
        # Tahminleri MIDI ve WAV formatına dönüştür
        json_to_midi(Hiperparameters.predicted_track_path, predicted_notes_midi)
        quantize_midi_file(predicted_notes_midi, predicted_notes_midi, quantize_to=Hiperparameters.quantize_to)
        combine_midi_to_wav(predicted_notes_midi, Hiperparameters.generated_wav)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
