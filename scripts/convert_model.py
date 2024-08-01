import argparse
import os
import requests
from transformers import Wav2Vec2ForCTC
from safetensors import safe_open
import tensorflow as tf


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def convert_model(safetensors_path, output_path):
    # Load the safetensors model
    with safe_open(safetensors_path, framework="pt") as f:
        state_dict = {k: v.numpy() for k, v in f.items()}

    # Convert to TensorFlow format
    model = Wav2Vec2ForCTC.from_pretrained(
        safetensors_path, state_dict=state_dict, from_safetensors=True
    )

    # Export to TensorFlow SavedModel format
    tf_model = model.to_tf_model()
    tf_model.save_pretrained(output_path)

    # Convert the TensorFlow model to TensorFlow.js format
    os.system(
        f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {output_path} {output_path}/tfjs_model"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a safetensors model to TensorFlow.js format."
    )
    parser.add_argument(
        "--url", required=True, help="HTTPS URL of the safetensors model file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the converted TensorFlow.js model"
    )
    args = parser.parse_args()

    safetensors_path = os.path.join(args.output, "model.safetensors")
    download_file(args.url, safetensors_path)
    convert_model(safetensors_path, args.output)


if __name__ == "__main__":
    main()
