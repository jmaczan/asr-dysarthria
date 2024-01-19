import os
import csv
import shutil
import argparse
from collections import deque


def build_dataset(
    start_path,
    output,
):
    visited = set()
    queue = deque([start_path])

    while queue:
        current_path = queue.popleft()
        if current_path in visited:
            continue

        visited.add(current_path)
        for entry in os.scandir(current_path):
            if entry.is_dir():
                if entry.name.startswith("Session"):
                    process_session_directory(entry.path, output, "metadata.csv")
                queue.append(entry.path)


def process_session_directory(session_path, output, metadata_file):
    wav_path = os.path.join(session_path, "wav_arrayMic")
    prompts_path = os.path.join(session_path, "prompts")

    if not os.path.exists(wav_path) or not os.path.exists(prompts_path):
        return

    for wav_file in os.listdir(wav_path):
        if wav_file.endswith(".wav"):
            new_wav_name = os.path.join(
                output, os.path.basename(session_path) + "_" + wav_file
            )

            txt_file = os.path.splitext(wav_file)[0] + ".txt"
            txt_file_path = os.path.join(prompts_path, txt_file)

            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r") as file:
                    content = file.read().strip()
                    with open(
                        metadata_file, "a", newline="", encoding="utf-8"
                    ) as csv_file:
                        if not content.startswith("[") and "input/" not in content:
                            content = content.replace('"', "")
                            shutil.copy(os.path.join(wav_path, wav_file), new_wav_name)
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

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if input_path is not None and output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        build_dataset(input_path, output_path)
        print("Done 😺")
    else:
        parser.print_help()
