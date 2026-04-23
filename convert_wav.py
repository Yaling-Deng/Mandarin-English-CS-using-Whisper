import librosa
import soundfile as sf   
import os
from tqdm import tqdm

wav_list = os.listdir("wav")

for i in tqdm(range(len(wav_list))):
    file_path = wav_list[i]
    audio, sample_rate = librosa.load("wav/"+file_path)
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    sf.write("wav2/"+file_path, audio, 16000)
    