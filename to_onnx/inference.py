import numpy as np
import onnxruntime
import soundfile as sf
from transformers import AutoProcessor

# Load the ONNX model
available_providers = onnxruntime.get_available_providers()
print("Available providers:", available_providers)

# Prefer CoreML, then fall back to CPU
preferred_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# Find the first available preferred provider
provider = next((p for p in preferred_providers if p in available_providers), None)

if provider:
    print(f"Using provider: {provider}")
    ort_session = onnxruntime.InferenceSession("model.onnx", providers=[provider])
else:
    print("No preferred provider available. Using default.")
    ort_session = onnxruntime.InferenceSession("model.onnx")

ort_session = onnxruntime.InferenceSession("model.onnx")
print("Provider being used:", ort_session.get_providers())
# Load the processor
processor = AutoProcessor.from_pretrained(
    "jmaczan/wav2vec2-large-xls-r-300m-dysarthria"
)


def transcribe(audio_file):
    # Load audio
    speech, sample_rate = sf.read(audio_file)

    # Resample if necessary
    if sample_rate != 16000:
        print(
            "Warning: audio sample rate is not 16kHz. Resampling might be necessary for accurate results."
        )

    # Preprocess the audio
    inputs = processor(
        speech, sampling_rate=sample_rate, return_tensors="np", padding=True
    )

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: inputs.input_values}
    logits = ort_session.run(None, ort_inputs)[0]
    print(logits.shape)
    # Post-process the output
    predicted_ids = np.argmax(logits, axis=-1)
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
