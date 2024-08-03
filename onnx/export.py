import torch.onnx


def export_to_onnx(model, processor, onnx_path):
    # Prepare a dummy input
    dummy_input = torch.randn(1, 16000)  # Adjust the input shape as needed

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence"},
            "output": {0: "batch_size", 1: "sequence_out"},
        },
    )

    print(f"Model exported to {onnx_path}")


onnx_path = "wav2vec2_model.onnx"
export_to_onnx(model, processor, onnx_path)
