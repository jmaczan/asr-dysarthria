### TODO
model = Wav2Vec2ForCTC.from_pretrained(repo_name).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(repo_name)

input_dict = processor(
    common_voice_test[0]["input_values"], return_tensors="pt", padding=True
)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

df = load_parquet_directory(directory_path, use_narrow_data=True)
dataset = convert_dataframe_to_dataset(df)

common_voice_test_transcription = dataset.train_test_split(test_size=0.2)["test"]

print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(common_voice_test_transcription[0]["transcription"].lower())
