from onnxruntime.transformers import optimizer
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from to_onnx.export_to_onnx import export_to_onnx_model
from to_onnx.load_model import load_or_create_model


def get_model_params(model):
    config = model.config
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Number of hidden layers: {config.num_hidden_layers}")
    print(f"Intermediate size: {config.intermediate_size}")

    return config


def optimize_onnx_model(onnx_path="model.onnx", optimized_path="optimized_model.onnx"):

    opt_model_path = optimizer.optimize_by_onnxruntime(
        onnx_path,
        optimized_model_path=optimized_path,
        model_type="default",
        num_heads=16,
        hidden_size=1024,
        optimization_options=None,
    )

    print(f"Optimized model saved to {opt_model_path}")


def quantize_onnx_model(optimized_path, quantized_path):
    quantize_dynamic(optimized_path, quantized_path, weight_type=QuantType.QInt8)
    print(f"Quantized model saved to {quantized_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="Path to input onnx file",
        required=False,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path of onnx file",
        default="optimized_model.onnx",
        required=False,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model HuggingFace name (optional), like jmaczan/wav2vec2-large-xls-r-300m-dysarthria. If provided, will load a model from HF",
        required=False,
    )

    args = parser.parse_args()
    print(args)
    input_path = args.input or "input_model.onnx"
    output_path = args.output
    model_id = args.model

    if model_id is not None:
        processor, model = load_or_create_model()
        export_to_onnx_model(model, input_path)
        config = get_model_params(model)  # temporary

    optimize_onnx_model(input_path, output_path)
    quantize_onnx_model(output_path, output_path)
