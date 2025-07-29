import onnxruntime
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

class BackgroundRemoverONNX:
    def __init__(self, model_path="rmbg.onnx", logger=None):
        # Use DirectML for GPU support if available, else use CPU
        self.session = onnxruntime.InferenceSession(model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.logger = logger if logger is not None else print
        self.logger_file = self.TqdmToLogger(self.logger) # For redirecting tqdm output

    def process_image(self, image_path, output_path="output.png", show_progress=True):
        try:
            self.logger(f"Processing: {image_path}") 
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            # Resize the image to the model's expected input size using LANCZOS for best quality
            image_resized = image.resize((1024, 1024), Image.LANCZOS)
            image_np = np.array(image_resized).astype(np.float32) / 255.0 # Normalize pixel values to [0 to 1]
            image_np = np.transpose(image_np, (2, 0, 1)) # HWC -> CHW

            # Normalize the image with mean and std 
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
            image_np = (image_np - mean) / std
            input_tensor = np.expand_dims(image_np, axis=0) # shape = 1x3x1024x1024

            if show_progress:
                with tqdm(total=3, desc="Processing Image", unit="step", file=self.logger_file) as pbar: 
                    pbar.set_description("Removing background")
                    result = self.session.run([self.output_name], {self.input_name: input_tensor})[0] # pass image to model
                    pbar.update(1)

                    pbar.set_description("Creating mask")
                    # Post-process the output to create a mask(gradience to decise which part of the image to keep) and resize to original image size
                    result = np.squeeze(result)
                    mask = (result * 255).astype(np.uint8)
                    mask_image = Image.fromarray(mask).resize(original_size, Image.LANCZOS)
                    pbar.update(1)

                    # Apply the mask to the original image and save
                    pbar.set_description("Applying mask and saving")
                    original_image = Image.open(image_path).convert("RGBA")
                    original_image.putalpha(mask_image)
                    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                    original_image.save(output_path)
                    pbar.update(1)
                    self.logger("Complete!!") 
            else:
                self.logger(f"Processing: {image_path}") 
                result = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
                result = np.squeeze(result)
                mask = (result * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask).resize(original_size, Image.LANCZOS)
                original_image = Image.open(image_path).convert("RGBA")
                original_image.putalpha(mask_image)
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                original_image.save(output_path)

            self.logger(f"Background removed! Saved to {output_path}") 
            return output_path

        except Exception as e:
            self.logger(f"Error processing {image_path}: {e}") 
            return None

    # Process multiple images in a folder 
    def process_batch(self, input_folder, output_folder, extensions=('.jpg', '.jpeg', '.png', '.webp')):
        if not os.path.exists(input_folder):
            self.logger(f"Input folder not found: {input_folder}")
            return []

        os.makedirs(output_folder, exist_ok=True)
        processed_files = []

        files_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith(extensions)]

        if not files_to_process:
            self.logger("No valid image files found in input folder") 
            return []

        self.logger(f"Found {len(files_to_process)} images to process") 

        for filename in tqdm(files_to_process, desc="Processing Images", unit="image", file=self.logger_file): 
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_no_bg.png"
            output_path = os.path.join(output_folder, output_filename)

            result = self.process_image(input_path, output_path, show_progress=False) # proceess image 
            if result:
                processed_files.append(result)

        self.logger(f"Processed {len(processed_files)} images successfully") 
        return processed_files

    # Helper class to redirect tqdm output to the logger
    class TqdmToLogger(object):
        def __init__(self, logger, level=None):
            self.logger = logger
            self.level = level if level is not None else 'info'
            self.buffer = ''

        def write(self, buf):
            self.buffer = buf.strip('\r\n\t ')
            if self.buffer:
                self.logger(self.buffer)

        def flush(self):
            pass

if __name__ == "__main__":
    remover = BackgroundRemoverONNX(model_path="rmbg.onnx")
    #Test case
    remover.process_image("image/test.webp", "output/result_onnx.png")