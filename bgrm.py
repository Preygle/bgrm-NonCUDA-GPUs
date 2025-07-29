from transformers import pipeline
from PIL import Image
import os
from tqdm import tqdm


class BackgroundRemover:
    def __init__(self, device="cpu", model_kwargs=None):
        self.pipe = None
        self.device = device
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    def _load_model(self):
        """Load the model only when needed (lazy loading)"""
        if self.pipe is None:
            print(f"Loading background removal model on device: {self.device}...")
            self.pipe = pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                use_fast=True,
                device=self.device,
                model_kwargs=self.model_kwargs
            )
            print("Model loaded successfully!")

    def process_image(self, image_path, output_path="output.png", show_progress=True):
        """Process a single image"""
        try:
            self._load_model()

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image not found: {image_path}")

            if show_progress:
                with tqdm(total=3, desc="Processing Image", unit="step") as pbar:
                    pbar.set_description("Loading image")
                    image = Image.open(image_path)
                    pbar.update(1)

                    pbar.set_description("Removing background")
                    result = self.pipe(image)
                    pbar.update(1)

                    pbar.set_description("Saving result")
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(
                        output_path) else ".", exist_ok=True)
                    result.save(output_path)
                    pbar.update(1)
                    pbar.set_description("Complete!")
            else:
                print(f"Processing: {image_path}")
                image = Image.open(image_path)
                result = self.pipe(image)
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(
                    output_path) else ".", exist_ok=True)
                result.save(output_path)

            print(f"Background removed! Saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_batch(self, input_folder, output_folder, extensions=('.jpg', '.jpeg', '.png', '.webp')):
        """Process multiple images in a folder with progress bar"""
        if not os.path.exists(input_folder):
            print(f"Input folder not found: {input_folder}")
            return []

        os.makedirs(output_folder, exist_ok=True)
        processed_files = []

        files_to_process = [f for f in os.listdir(
            input_folder) if f.lower().endswith(extensions)]

        if not files_to_process:
            print("No valid image files found in input folder")
            return []

        print(f"Found {len(files_to_process)} images to process")

        for filename in tqdm(files_to_process, desc="Processing Images", unit="image"):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_no_bg.png"
            output_path = os.path.join(output_folder, output_filename)

            result = self.process_image(
                input_path, output_path, show_progress=False)
            if result:
                processed_files.append(result)

        print(f"Processed {len(processed_files)} images successfully")
        return processed_files


remover_batch_gpu = BackgroundRemover(device="cpu") 
results = remover_batch_gpu.process_batch("image/", "output_images/")
