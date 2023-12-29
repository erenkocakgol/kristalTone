import configparser
import logging

class Hiperparameters():
    config = configparser.ConfigParser()
    config.read('config.cfg')

    try:
        # Reading hyperparameters and paths from config file
        epochs = int(config['Training']['epochs'])
        batch_size = int(config['Training']['batch_size'])
        learning_rate = float(config['Training']['learning_rate'])
        hidden_size = int(config['Training']['hidden_size'])
        num_layers = int(config['Training']['num_layers'])

        dataset_path = config['Paths']['dataset_path']
        test_midi_path = config['Paths']['test_midi_path']
        predicted_track_path = config['Paths']['predicted_track_path']
        generated_wav = config['Paths']['generated_wav_path']

        quantize_to = float(config['Quantization']['quantize_to'])

    except Exception as e:
        logging.error(f"An Hiperparameter error occurred: {e}")