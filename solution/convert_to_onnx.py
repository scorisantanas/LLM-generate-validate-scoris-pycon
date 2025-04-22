# -*- coding: utf-8 -*-
"""
This script demonstrates how to convert a pre-trained Hugging Face MarianMT
translation model (e.g., from English to Lithuanian) into the ONNX format
using the Optimum library. It then compares the output of the original
PyTorch model and the converted ONNX model on a sample text to verify
the conversion process.
"""

# --- Imports ---
import os                     # Used for operating system dependent functionalities like path joining and directory creation.
from transformers import MarianMTModel, MarianTokenizer, pipeline # Hugging Face Transformers library components: the base model, its tokenizer, and the easy-to-use pipeline function.
from optimum.onnxruntime import ORTModelForSeq2SeqLM # Optimum library class specifically for handling ONNX Runtime Sequence-to-Sequence models.
# import torch # PyTorch is implicitly used by transformers and optimum, but not directly called in this simplified script.

# --- Configuration ---

# Set the path to your fine-tuned MarianMT model directory.
# This directory should contain files like 'pytorch_model.bin', 'config.json',
# 'tokenizer_config.json', 'vocab.json', 'source.spm', 'target.spm', etc.
SOURCE_MODEL_PATH = "path/to/your/huggingface/marian_model" # e.g., "models/opus-mt-en-lt-finetuned"

# Define the directory where the converted ONNX model will be saved.
# This will typically be a subdirectory within the source model path.
ONNX_EXPORT_PATH = os.path.join(SOURCE_MODEL_PATH, "onnx_exported")

# Ensure the target directory for the ONNX model exists.
os.makedirs(ONNX_EXPORT_PATH, exist_ok=True)

# --- Load Tokenizer ---

# Load the tokenizer associated with the pre-trained MarianMT model.
# The tokenizer handles converting text into numerical IDs that the model understands.
print(f"Loading tokenizer from: {SOURCE_MODEL_PATH}")
tokenizer = MarianTokenizer.from_pretrained(SOURCE_MODEL_PATH)
print("Tokenizer loaded.")

# --- Step 1: Export the Model to ONNX Format ---

# Use the Optimum library to load the Transformers model and convert it to ONNX format.
# `from_transformers=True` tells Optimum to load the model from a standard Transformers checkpoint.
print(f"Exporting the model from '{SOURCE_MODEL_PATH}' to ONNX format...")
onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
    SOURCE_MODEL_PATH,
    from_transformers=True
)

# Save the converted ONNX model and associated configuration files to the specified path.
print(f"Saving the ONNX model to: {ONNX_EXPORT_PATH}")
onnx_model.save_pretrained(ONNX_EXPORT_PATH)
print("ONNX model exported and saved successfully.")

# --- Step 2: Test Translation with the ONNX Model ---

print("\nTesting translation with the exported ONNX model...")

# Define a sample text string for translation.
sample_text = "The return on equity reached an impressive level, indicating effective management."

# Create a translation pipeline using the loaded ONNX model and tokenizer.
# Pipelines provide a convenient way to perform inference.
# `device=-1` forces the pipeline to run on the CPU. Change to `device=0` for the first GPU.
# Ensure the task name `translation_en_to_lt` matches your model's language pair.
onnx_translator = pipeline(
    "translation_en_to_lt", # Adjust task if your model translates different languages (e.g., "translation_fr_to_en")
    model=onnx_model,       # Use the ONNX model loaded via Optimum
    tokenizer=tokenizer,    # Use the same tokenizer as the original model
    device=-1               # Use CPU (-1) or GPU (0, 1, ...)
)

# Perform the translation using the ONNX pipeline.
onnx_translation_result = onnx_translator(sample_text)
onnx_translation_text = onnx_translation_result[0]['translation_text'] # Extract the translated text

print(f"ONNX model translation: {onnx_translation_text}")

# --- Step 3: Test Translation with the Original PyTorch Model ---

print("\nTesting translation with the original PyTorch model for comparison...")

# Load the original PyTorch MarianMT model from the source path.
print(f"Loading original PyTorch model from: {SOURCE_MODEL_PATH}")
pytorch_model = MarianMTModel.from_pretrained(SOURCE_MODEL_PATH)
print("Original PyTorch model loaded.")

# Create a similar translation pipeline, but this time using the original PyTorch model.
pytorch_translator = pipeline(
    "translation_en_to_lt", # Adjust task if your model translates different languages
    model=pytorch_model,    # Use the original PyTorch model
    tokenizer=tokenizer,    # Use the same tokenizer
    device=-1               # Use CPU (-1) or GPU (0, 1, ...)
)

# Perform the translation using the PyTorch pipeline.
pytorch_translation_result = pytorch_translator(sample_text)
pytorch_translation_text = pytorch_translation_result[0]['translation_text'] # Extract the translated text

print(f"PyTorch model translation: {pytorch_translation_text}")

# --- Step 4: Compare the Translations ---

print("\nComparing ONNX and PyTorch translations...")

# Compare the output strings, ignoring potential leading/trailing whitespace differences.
if onnx_translation_text.strip() == pytorch_translation_text.strip():
    print("✅ Success: The ONNX model's translation matches the PyTorch model's translation.")
else:
    # If translations differ, print both for inspection. Minor differences might occur due
    # to floating-point precision variations during conversion, but significant differences
    # could indicate a problem in the export process.
    print("⚠️ Warning: The translations differ.")
    print(f"  ONNX Output:    '{onnx_translation_text}'")
    print(f"  PyTorch Output: '{pytorch_translation_text}'")

print("\nScript finished.")