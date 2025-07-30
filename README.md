
# AI Background Remover

[![GitHub stars](https://img.shields.io/github/stars/your-username/your-repo.svg?style=social&label=Star)](https://github.com/Preygle/bgrm-NonCUDA-GPUs/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/your-username/your-repo.svg?style=social&label=Fork)](https://github.com/Preygle/bgrm-NonCUDA-GPUs/network/members)

Ever wanted to yeet the background out of your images with the power of AI? Look no further! This project uses the incredible [BriaAI RMBG-1.4 model](https://huggingface.co/briaai/RMBG-1.4) to automagically remove backgrounds from your images. It's fast, it's easy to use, and it's powered by the magic of ONNX for some serious performance gains.

Whether you're a developer looking to integrate background removal into your app, or just someone who wants to make some fire memes, this project has got you covered.

## Features

*   **Single Image Processing:** Remove the background from a single image with a simple command.
*   **Batch Processing:** Have a whole folder of images? No problem! Process them all in one go.
*   **ONNX Powered:** We use the ONNX runtime for high-performance inference, so you can get your background-free images in a flash.
*   **GUI Included:** Don't like the command line? We've got you covered with a user-friendly GUI.
*   **Hugging Face Integration:** We use the latest and greatest models from the Hugging Face Hub.

## Getting Started

Ready to dive in? Here's how to get started:

### 1. Clone the Repo

```bash
git https://github.com/Preygle/bgrm-NonCUDA-GPUs.git
cd bgrm-NonCUDA-GPUs
```

### 2. Install Dependencies

We've made it super easy to install all the necessary dependencies. Just run this command:

```bash
pip install -r requirements.txt
```

### 3. Download the Model

This project uses the BriaAI RMBG-1.4 model, which is available on Hugging Face.
Follow this link to download the model [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4/tree/main/onnx)
## 👨‍💻 Usage

We've provided a few different ways to use this project, so you can choose the one that best suits your needs.

### Python Scripts

We have a few different Python scripts that you can use to remove backgrounds from your images.

#### `bgrm.py`

This Python script uses the transformers pipeline with the briaai/RMBG-1.4 model to remove image backgrounds, supporting both single and batch processing (Current code force runs on CPU).

```bash
python bgrm.py
```

#### `bgrm_onnx.py`

This script uses the ONNX runtime for high-performance inference. It's a great option if you need to process a lot of images quickly.

```bash
python bgrm_onnx.py
```

###  GUI

If you're not a fan of the command line, we've got you covered with a user-friendly GUI. To launch it, just run this command:

```bash
python bgrm_ui.py
```

This will open up a window where you can select your input and output folders, and then process your images with the click of a button.

## The Model

This project uses the [BriaAI RMBG-1.4 model](https://huggingface.co/briaai/RMBG-1.4), which is a state-of-the-art background removal model. It's been trained on a massive dataset of images, so it can handle a wide variety of subjects and scenes.

## Contributing

We're always looking for new contributors! If you have an idea for a new feature or a bug fix, please open an issue or submit a pull request.


