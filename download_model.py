import requests
import os
from tqdm import tqdm

url = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/rmbg.onnx"
file_path = "rmbg.onnx"

if not os.path.exists(file_path):
    print(f"Downloading ONNX model from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_path) as pbar:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)

        print("Model downloaded successfully!")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during model download: {e}")
else:
    print(f"ONNX model already exists at {file_path}. Skipping download.")
