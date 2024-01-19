import os
import csv
import argparse


def build_dataset(input_directory, transcriptions_dir, manifest_file):
    audio_files = [f for f in os.listdir(input_directory) if f.endswith(".wav")]

    with open(manifest_file, "w", newline="") as csvfile:
        manifest_writer = csv.writer(csvfile)
        manifest_writer.writerow(["file_name", "transcription"])

        for audio_file in audio_files:
            base_filename = os.path.splitext(audio_file)[0]
            transcription_file = base_filename + ".txt"

            if transcription_file in os.listdir(transcriptions_dir):
                audio_path = os.path.join(input_directory, audio_file)
                transcription_path = os.path.join(
                    transcriptions_dir, transcription_file
                )

                with open(transcription_path, "r", encoding="utf-8") as f:
                    transcription_content = f.read().strip()

                if (
                    not transcription_content.startswith("[")
                    and "input/" not in transcription_content
                ):
                    clean_transcription_content = transcription_content.replace('"', "")
                    manifest_writer.writerow([audio_path, clean_transcription_content])
                else:
                    file_path = os.path.join("data", audio_file)

                    if os.path.exists(file_path):
                        os.remove(file_path)

            else:
                print(f"Transcription file for {audio_file} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument(
        "--input", type=str, help="Path to the original TORGO dataset directory"
    )
    parser.add_argument(
        "--output", type=str, help="This is where parsed dataset will land"
    )

    args = parser.parse_args()

    directory_path = args.directory
    output_path = args.output

    # Your code here using directory_path and output_path

    audio_dir = "data"
    transcriptions_dir = "transcriptions"
    manifest_file = "metadata.csv"

    build_dataset(audio_dir, transcriptions_dir, manifest_file)
    print("Dataset created successfully")
