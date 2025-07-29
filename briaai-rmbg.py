from gradio_client import Client
import time


def remove_background_space(image_path):
    try:
        # Try the direct space URL first
        client = Client("Xenova/remove-background-webgpu")
        result = client.predict(image_path, api_name="/predict")
        return result
    except Exception as e:
        print(f"Error with direct space: {e}")
        # Fallback to full URL
        try:
            client = Client(
                "https://huggingface.co/spaces/Xenova/remove-background-webgpu")
            result = client.predict(image_path, api_name="/predict")
            return result
        except Exception as e2:
            print(f"Error with full URL: {e2}")
            return None


# Usage
result = remove_background_space("image/test.webp")
if result:
    print(f"Success! Result: {result}")
else:
    print("Failed to process image")
