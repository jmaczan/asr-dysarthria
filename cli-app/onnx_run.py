import numpy as np
import onnxruntime
import soundfile as sf
from transformers import AutoProcessor

# Load the ONNX model
ort_session = onnxruntime.InferenceSession("model.onnx")

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

    # Post-process the output
    predicted_ids = np.argmax(logits, axis=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]


# Example usage
audio_file = "file.wav"
result = transcribe(audio_file)
print(f"Transcription: {result}")
