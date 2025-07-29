import requests
from PIL import Image
import io


def remove_background_api(image_path, output_path="output.png"):
    # API endpoint for the same model used in Xenova space
    API_URL = "https://api-inference.huggingface.co/models/briaai/RMBG-1.4"

    # Optional: Add HuggingFace token for better rate limits
    # Get token from huggingface.co/settings/tokens
    headers = {"Authorization": "Bearer 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"}

    try:
        # Read and send the image
        with open(image_path, "rb") as f:
            data = f.read()

        response = requests.post(API_URL, headers=headers, data=data)

        if response.status_code == 200:
            # Save the result
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Background removed! Saved to {output_path}")
            return output_path
        else:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


# Usage
result = remove_background_api("image/test.webp", "output_no_bg.png")
