from scipy.io import wavfile
from preprocess_sound import preprocess_sound
from numpy.random import seed, randint
import matplotlib.pyplot as plt

seg_len = 5
seg_num = 60

sound_file = '72021001.wav'
sr, wav_data = wavfile.read(sound_file)

length = sr * seg_len           # 5s segment
range_high = len(wav_data) - length
seed(1)  # for consistency and replication
random_start = randint(range_high, size=seg_num)

cur_wav = wav_data[random_start[0]:random_start[0] + length]
cur_wav = cur_wav / 32768.0
cur_spectro = preprocess_sound(cur_wav, sr)

num_rows = 1000
num_columns = len(wav_data) // num_rows
reshaped_wav_data = wav_data[:num_rows * num_columns].reshape(num_rows, num_columns)


print("Shape of wav_data:", wav_data.shape)
print("Shape of cur_spectro:", cur_spectro.shape)

# Plot reshaped wav_data
plt.figure(figsize=(10, 4))
plt.imshow(reshaped_wav_data, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Samples')
plt.ylabel('Segments')
plt.title('2D Visualization of wav_data')
plt.show()

# Select the first element of cur_spectro to get a 2D array
spectro_2d = cur_spectro[0]

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(spectro_2d, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Intensity (dB)')
plt.xlabel('Time Bins')
plt.ylabel('Frequency Bins')
plt.title('Spectrogram Visualization')
plt.show()

print(wav_data)
print(cur_spectro)
