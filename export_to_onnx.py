
import torch
from transformers import AutoModel, AutoImageProcessor
import os

def export_model_to_onnx(model_name="briaai/RMBG-1.4", output_path="rmbg.onnx"):
    """
    Loads a PyTorch model from Hugging Face and exports it to the ONNX format.
    """
    if os.path.exists(output_path):
        print(f"ONNX model already exists at {output_path}. Skipping export.")
        return

    print("Loading the model and image processor from Hugging Face...")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Set the model to evaluation mode
    model.eval()

    print("Creating a dummy input for ONNX export...")
    # The model expects a batch of images. We create a dummy input with dynamic axes 
    # to allow for variable batch size and image dimensions.
    dummy_input = torch.randn(1, 3, 1024, 1024) # Batch size 1, 3 channels, 1024x1024 resolution

    print(f"Exporting the model to ONNX format at: {output_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=14, # A reasonably modern opset version
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print("Model exported successfully!")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")


if __name__ == "__main__":
    # Before running, ensure you have the required packages:
    # pip install torch transformers onnx
    export_model_to_onnx()
