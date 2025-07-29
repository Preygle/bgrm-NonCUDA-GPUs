Usage Examples

1. Default usage (CPU)
remover = BackgroundRemover()
result = remover.process_image("image/test.webp", "output/result.png")


2. Use a specific GPU (e.g., the first CUDA device)
Make sure you have PyTorch with CUDA support installed
remover_gpu = BackgroundRemover(device="cuda:0")
result_gpu = remover_gpu.process_image("image/test.webp", "output/result_gpu.png")


3. Use CPU with 8-bit quantization to reduce memory usage
remover_quantized = BackgroundRemover(device="cpu", model_kwargs={"load_in_8bit": True})
result_quantized = remover_quantized.process_image("image/test.webp", "output/result_quantized.png")


4. Batch processing on a specific GPU
remover_batch_gpu = BackgroundRemover(device="cpu") # Change to "cuda:0" if you have a GPU
results = remover_batch_gpu.process_batch("image/", "output_images/")