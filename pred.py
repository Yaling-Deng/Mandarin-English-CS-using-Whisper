import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset
import librosa
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "whisper-small-res/checkpoint-4000"
model_id = "whisper-large-res"

pipe = pipeline(
  "automatic-speech-recognition",
  model=model_id,
#   tokenizer="whisper-small",
  chunk_length_s=30,
  device=device,
#   language="zh",
)

# file_path = "wav2/_1634_210_2577_1_1525157964032_3712259_12.wav"


# print(prediction)

with open("test_set/label.txt", "r") as f:
    lines = f.readlines()

new_lines = []
for line in tqdm(lines,total=len(lines)):
    # print(line)
    file_name = line.split(" ")[0]
    file_path = "wav2/" + file_name + ".wav"
    # print(file_path)

    audio, sample_rate = librosa.load(file_path)
    prediction = pipe(audio, batch_size=8)["text"]
    new_lines.append(file_name + " " + prediction + "\n")

    # break

with open('test_set/output_fine_large.txt', 'w') as f:
    for item in new_lines:
        f.write("%s" % item)