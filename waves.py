import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def sine_wave(frequency, duration, sample_rate=44100):
    if frequency <= 0 or duration <= 0:
        logging.error("Invalid parameters for sine wave. Frequency and duration must be positive.")
        return None

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)
    return wave

def square_wave(frequency, duration, sample_rate=44100):
    if frequency <= 0 or duration <= 0:
        logging.error("Invalid parameters for square wave. Frequency and duration must be positive.")
        return None

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sign(np.sin(2 * np.pi * frequency * t))
    return wave

# Usage example (commented for direct script execution)
# sine_wave_audio = sine_wave(frequency=440, duration=1)  # 440 Hz for 1 second
# square_wave_audio = square_wave(frequency=440, duration=1)  # 440 Hz for 1 second
