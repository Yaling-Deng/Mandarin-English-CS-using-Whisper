from datasets import load_dataset,Dataset
from transformers import WhisperFeatureExtractor
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)
# from datasets import Audio
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
# import evaluate
from wer import WER
# import torchaudio
import librosa
import soundfile as sf


# model_path = "whisper-small"
model_path = "whisper-large-v3"

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 去除每行末尾的换行符
    lines = [{"content": line.strip()} for line in lines]
    return lines

# 文件路径
train_file_path = 'train_set/label.txt'  # 将此路径替换为你的TXT文件的实际路径
test_file_path = 'test_set/label.txt'

# 读取TXT文件内容
train_data = read_txt_file(train_file_path)
train_dataset = Dataset.from_list(train_data)

test_data = read_txt_file(test_file_path)
test_dataset = Dataset.from_list(test_data)

print(train_dataset[0])
print(len(train_dataset))
# exit()

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_path, language="zh", task="transcribe"
)

processor = WhisperProcessor.from_pretrained(
    model_path, language="zh", task="transcribe"
)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    # audio = batch["audio"]
    # new_examples = []
    # for batch in examples:
    content = batch["content"]
    # import pdb
    # pdb.set_trace()
    content_list = content.split(" ")
    file_path = content_list[0] + ".wav"
    sentence = " ".join(content_list[1:])

    # audio, sample_rate = torchaudio.load("wav/"+file_path)
    audio, sample_rate = librosa.load("wav2/"+file_path)
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    # import pdb
    # pdb.set_trace()

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio, sampling_rate=16000).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(sentence).input_ids
    # new_examples.append(batch)
    return batch
    # return new_examples

train_dataset = train_dataset.shard(num_shards=100, index=0)
test_dataset = test_dataset.shard(num_shards=100, index=0)

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)
# exit()

model = WhisperForConditionalGeneration.from_pretrained(model_path)

model.generation_config.language = "zh"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = WER()
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-res",
    # output_dir="./whisper-small-res",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

processor.save_pretrained(training_args.output_dir)
model.save_pretrained(training_args.output_dir)
