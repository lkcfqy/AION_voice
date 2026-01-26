import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
# import matplotlib.pyplot as plt
from src.voice_body.cochlea import Cochlea

DURATION = 3
SR = 16000

print(f"Recording for {DURATION} seconds at {SR}Hz...")
recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
sd.wait()
print("Recording complete.")

# Save raw
wav.write("debug_mic_input.wav", SR, (recording * 32767).astype(np.int16))
print("Saved debug_mic_input.wav")

# Process
cochlea = Cochlea(sample_rate=SR)
# Flatten for processing
audio_data = recording.flatten()
spec = cochlea.process(audio_data)

print(f"Spectrogram shape: {spec.shape}")
print(f"Spectrogram stats: Min={spec.min():.2f}, Max={spec.max():.2f}, Mean={spec.mean():.2f}")

# Plot
# plt.imshow(spec[:,:,0], origin='lower', aspect='auto')
# plt.title("Debug Spectrogram")
# plt.savefig("debug_spectrogram.png")
# print("Saved debug_spectrogram.png")
