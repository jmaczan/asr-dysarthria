import os
import csv
import shutil
import argparse
from collections import deque


def remove_prefix(a, b):
    if b.startswith(a):
        return b[len(a) :]
    return b


def build_dataset(input_path, output_path, **kwargs):
    visited = set()
    queue = deque([input_path])

    metadata_path = os.path.join(
        output_path, kwargs.get("metadata_file", "metadata.csv")
    )

    with open(metadata_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "transcription"])

    while queue:
        current_path = queue.popleft()
        if current_path in visited:
            continue

        visited.add(current_path)
        for entry in os.scandir(current_path):
            if entry.is_dir():
                if entry.name.startswith("Session"):
                    prefix = (
                        os.path.basename(os.path.dirname(os.path.dirname(entry)))
                        + "_"
                        + os.path.basename(os.path.dirname(entry))
                        + "_"
                    )
                    process_session_directory(
                        entry.path,
                        output_path,
                        prefix,
                        **kwargs,
                    )
                queue.append(entry.path)


def process_session_directory(
    single_input_path, output_path, wav_file_prefix="", **kwargs
):
    metadata_path = os.path.join(
        output_path, kwargs.get("metadata_file", "metadata.csv")
    )

    huggingface_format = kwargs.get("huggingface_format", False)

    audio_output_path = kwargs.get("audio_output_path", output_path)

    wav_path = os.path.join(single_input_path, "wav_arrayMic")
    prompts_path = os.path.join(single_input_path, "prompts")

    if not os.path.exists(wav_path) or not os.path.exists(prompts_path):
        return

    for wav_file in os.listdir(wav_path):
        if wav_file.endswith(".wav"):
            basic_wav_name = (
                wav_file_prefix + os.path.basename(single_input_path) + "_" + wav_file
            )

            new_wav_name = (
                os.path.join(basic_wav_name)
                if not huggingface_format
                else os.path.join("data", basic_wav_name)
            )

            wav_output_path = os.path.join(audio_output_path, basic_wav_name)

            txt_file = os.path.splitext(wav_file)[0] + ".txt"
            txt_file_path = os.path.join(prompts_path, txt_file)

            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r") as file:
                    content = file.read().strip()
                    with open(
                        metadata_path, "a", newline="", encoding="utf-8"
                    ) as csv_file:
                        if not content.startswith("[") and "input/" not in content:
                            content = content.replace('"', "")
                            shutil.copy(
                                os.path.join(wav_path, wav_file), wav_output_path
                            )
                            writer = csv.writer(csv_file)
                            writer.writerow([new_wav_name, content])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a dataset for ASR training using TORGO dataset. Code: https://github.com/jmaczan/asr-dysarthria. The original TORGO database: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html"
    )
    parser.add_argument(
        "--input", type=str, help="Path to the original TORGO dataset directory"
    )

    parser.add_argument(
        "--output", type=str, help="This is where parsed dataset will land"
    )

    parser.add_argument(
        "--clear-output-dir",
        action="store_true",
        help="(Optional) Remove existing content of output directory",
    )

    parser.add_argument(
        "--huggingface-format",
        action="store_true",
        help="(Optional) Put output wav files into additional data directory",
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    clear_output_directory = args.clear_output_dir
    huggingface_format = args.huggingface_format

    if input_path is None or output_path is None:
        parser.print_help()
        exit()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if clear_output_directory:
        print("Existing content of output directory is being removed")
        if os.path.exists(output_path):
            for filename in os.listdir(output_path):
                file_path = os.path.join(output_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    output_data_path = os.path.join(output_path, "data")

    if huggingface_format and not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
        audio_output_path = output_data_path
    else:
        audio_output_path = None

    build_dataset(
        input_path,
        output_path,
        audio_output_path=audio_output_path,
        clear_output_directory=clear_output_directory,
        huggingface_format=huggingface_format,
    )

    print("Dataset build done 😺")
