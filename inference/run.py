import os
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC

MODEL_DIR = "cached_model"
MODEL_NAME = "jmaczan/wav2vec2-large-xls-r-300m-dysarthria"


def load_or_create_model():
    if os.path.exists(MODEL_DIR):
        print("Loading model from cache...")
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        model = AutoModelForCTC.from_pretrained(MODEL_DIR)
    else:
        print("Downloading and caching model...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForCTC.from_pretrained(MODEL_NAME)

        # Save the model and processor
        processor.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)

    return processor, model


# Load the model and processor
processor, model = load_or_create_model()

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Inference using {device}")
model = model.to(device)


# Load and preprocess the audio file
def load_audio(file_path):
    speech, sample_rate = torchaudio.load(file_path)
    speech = speech.squeeze().numpy()
    return speech, sample_rate


# Perform inference
def transcribe(file_path):
    speech, sample_rate = load_audio(file_path)
    inputs = processor(
        speech, sampling_rate=sample_rate, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]


# Example usage
audio_file = "file.wav"
import time

start = time.time()
result = transcribe(audio_file)
end = time.time()
print(end - start)
print(f"Transcription: {result}")
