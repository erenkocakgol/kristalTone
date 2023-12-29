import pretty_midi
import numpy as np
import soundfile as sf
import os
import waves  # waves.py dosyanız

def midi_to_wav(midi_file_path, sample_rate=44100):
    # MIDI dosyasını yükle
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    total_duration = midi_data.get_end_time()

    # Ses dalgası dizisini başlat
    audio_wave = np.zeros(int(sample_rate * total_duration))

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Nota frekansını hesapla
            frequency = pretty_midi.note_number_to_hz(note.pitch)

            # Nota süresini hesapla
            start_sample = int(note.start * sample_rate)
            end_sample = int(note.end * sample_rate)
            duration = note.end - note.start

            # Dalga formunu üret
            wave = waves.sine_wave(frequency, duration, sample_rate)
            wave *= (note.velocity / 127)  # Velocity'yi normalize et

            # Dalga formunu ses dizisine ekle
            wave_length = end_sample - start_sample
            if len(wave) > wave_length:
                wave = wave[:wave_length]
            elif len(wave) < wave_length:
                wave = np.pad(wave, (0, wave_length - len(wave)), 'constant')

            audio_wave[start_sample:end_sample] += wave

    # Ses dalgasını normalize et
    max_val = np.max(np.abs(audio_wave))
    if max_val > 0:
        audio_wave /= max_val

    return audio_wave

def combine_midi_to_wav(input_directory, output_wav_file, sample_rate=44100):
    combined_wave = None

    for file in os.listdir(input_directory):
        if file.endswith('.mid') or file.endswith('.midi'):
            midi_file_path = os.path.join(input_directory, file)
            audio_wave = midi_to_wav(midi_file_path, sample_rate)

            if combined_wave is None:
                combined_wave = audio_wave
            else:
                # Ses dalgalarını uzunluğa göre doldur ve topla
                max_length = max(len(combined_wave), len(audio_wave))
                combined_wave = np.pad(combined_wave, (0, max_length - len(combined_wave)), 'constant')
                audio_wave = np.pad(audio_wave, (0, max_length - len(audio_wave)), 'constant')
                combined_wave += audio_wave

    # Son ses dalgasını normalize et
    max_val = np.max(np.abs(combined_wave))
    if max_val > 0:
        combined_wave /= max_val

    # WAV olarak kaydet
    sf.write(output_wav_file, combined_wave, sample_rate)

# Kullanımı
# combine_midi_to_wav('path/to/midi_files', 'combined_output.wav')
