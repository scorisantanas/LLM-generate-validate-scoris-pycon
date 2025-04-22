# -*- coding: utf-8 -*-
"""
This script continuously fetches text data from a database table, translates it 
from English ('en') to Lithuanian ('lt') using an ONNX-optimized transformer model, 
and updates the translated text back into the database.
"""

# --- Imports ---
import time         # Used for pausing execution (e.g., polling interval)
import re           # Used for regular expression operations (sentence splitting)
import torch        # PyTorch library, used for tensor operations and device management
import onnxruntime as ort # ONNX Runtime for running the optimized model
import pyodbc       # Library for connecting to ODBC databases (like SQL Server, etc.)
from transformers import MarianTokenizer          # Tokenizer for the MarianNMT model family
from optimum.onnxruntime import ORTModelForSeq2SeqLM # Optimum library class for ONNX Seq2Seq models

# --- Configuration ---

# Database connection string placeholder.
# Replace this with your actual database connection string.
# Example for SQL Server:
# DB_CONNECTION_STRING = (
#     "Driver={ODBC Driver 17 for SQL Server};"
#     "Server=your_server_name;"
#     "Database=your_database_name;"
#     "UID=your_username;"
#     "PWD=your_password;"
# )
DB_CONNECTION_STRING = "YOUR_DATABASE_CONNECTION_STRING"

# Database table and column names placeholders.
# Replace these with your actual table and column names.
SOURCE_TABLE = "YOUR_SOURCE_TABLE_NAME" # e.g., 'llm_finance_descriptions'
ID_COLUMN = "YOUR_ID_COLUMN"            # e.g., 'ID'
SOURCE_TEXT_COLUMN = "YOUR_SOURCE_TEXT_COLUMN" # e.g., 'generated_description_en'
TARGET_TEXT_COLUMN = "YOUR_TARGET_TEXT_COLUMN" # e.g., 'generated_description_lt'

# Path to the local ONNX model directory.
# Replace this with the actual path where your ONNX model is saved.
# This directory should contain files like 'encoder_model.onnx', 'decoder_model.onnx',
# 'decoder_with_past_model.onnx', 'config.json', 'vocab.json', 'merges.txt', etc.
LOCAL_MODEL_PATH = "path/to/your/local/onnx_model" # e.g., "models/opus-mt-en-lt/onnx"

# Processing batch sizes
DB_FETCH_BATCH_SIZE = 50   # How many records to fetch and update in the database at once
TRANSLATION_BATCH_SIZE = 128 # How many sentences to translate simultaneously by the model

# --- Setup ---

# Configure ONNX Runtime logging level (optional: 3 = Warning, 2 = Info, 1 = Verbose, 0 = Error)
ort.set_default_logger_severity(3)

# Determine execution device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
print(f"Using device: {device}. Provider: {provider}")

print(f'Loading translation model from: {LOCAL_MODEL_PATH}...')

# Load the tokenizer associated with the translation model
tokenizer_trans = MarianTokenizer.from_pretrained(LOCAL_MODEL_PATH)

# Set ONNX Runtime session options for potential performance improvements
# These options can be tuned based on the specific hardware and model
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Enable all available graph optimizations
sess_options.intra_op_num_threads = 8 # Number of threads for intra-operator parallelism
sess_options.inter_op_num_threads = 8 # Number of threads for inter-operator parallelism
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # Execute operators sequentially (can sometimes improve performance)

# Load the ONNX model for sequence-to-sequence tasks using Optimum
model_trans = ORTModelForSeq2SeqLM.from_pretrained(
    LOCAL_MODEL_PATH,
    provider=provider,          # Specify the execution provider (CPU or CUDA)
    session_options=sess_options, # Apply the custom session options
    use_cache=True              # Enable caching mechanism in the decoder (speeds up generation)
)

# Move the model to the selected device (CPU or GPU)
# Note: ORTModelForSeq2SeqLM handles device placement internally based on the provider,
# but explicit .to(device) is often good practice if interacting with PyTorch tensors directly.
# In this Optimum setup, it might not be strictly necessary but doesn't hurt.
model_trans.to(device)

print("Model loaded successfully.")

# --- Translation Functions ---

def batch_translate(texts_en, batch_size=TRANSLATION_BATCH_SIZE):
    """
    Translates a list of English texts to Lithuanian in batches.

    Args:
        texts_en (list): A list of English strings (sentences) to translate.
        batch_size (int): The number of sentences to process in each batch.

    Returns:
        list: A list of translated Lithuanian strings.
    """
    translated_texts = []
    # Process the texts in batches to manage memory usage
    for i in range(0, len(texts_en), batch_size):
        batch_texts = texts_en[i:i+batch_size]

        # Tokenize the batch: Convert text to numerical input IDs, pad to same length
        inputs = tokenizer_trans(
            batch_texts,
            return_tensors="pt",    # Return PyTorch tensors
            padding=True,           # Pad sequences to the longest sequence in the batch
            truncation=True,        # Truncate sequences longer than max model length
            max_length=256          # Maximum sequence length for the model
        )

        # Move input tensors to the designated device (CPU/GPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference without calculating gradients (saves memory and computation)
        with torch.no_grad():
            # Generate translations using the model
            # num_beams=1 uses greedy decoding (fastest)
            # use_cache=True leverages the key/value cache for faster generation
            translations = model_trans.generate(**inputs, num_beams=1, use_cache=True)

        # Move the generated translation tensors back to the CPU for decoding
        translations_cpu = translations.cpu()

        # Decode the numerical output IDs back into text strings
        batch_translations = tokenizer_trans.batch_decode(
            translations_cpu,
            skip_special_tokens=True # Remove special tokens like <pad>, </s>
        )

        # Clean up whitespace and add to the results list
        translated_texts.extend([text.strip() for text in batch_translations])

    return translated_texts

def translate_text_structure(text_en):
    """
    Translates English text while preserving paragraph structure.
    Splits text into paragraphs and sentences, translates sentences, and reconstructs.

    Args:
        text_en (str): The English text to translate.

    Returns:
        str: The translated Lithuanian text with paragraph structure preserved.
    """
    # Split the input text into paragraphs based on double newlines
    paragraphs = text_en.split('\n\n')
    translated_text_lt = ""

    for para in paragraphs:
        para_strip = para.strip()
        if not para_strip: # Skip empty paragraphs
            continue

        # Split paragraph into sentences. Using regex to avoid splitting on decimal points.
        # Looks for a period '.' not preceded or followed by a digit.
        sentences = re.split(r'(?<!\d)\.(?!\d)', para_strip)
        # Remove any empty strings resulting from the split and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        # Translate the sentences in the paragraph using the batching function
        translated_sentences = batch_translate(sentences)

        # Reconstruct the translated paragraph
        translated_para = ""
        for i, translated_sentence in enumerate(translated_sentences):
            # Ensure sentences end with a period (if the original likely did)
            original_ends_with_period = (i < len(sentences) and sentences[i].endswith('.'))
            # Add period if missing, crude check, might need refinement
            if not translated_sentence.endswith(('.', '!', '?')) and original_ends_with_period:
                 translated_sentence += '.'

            # Add the translated sentence to the paragraph string
            if translated_para: # Add space before appending next sentence
                translated_para += " "
            translated_para += translated_sentence

        # Append the translated paragraph to the final result, preserving paragraph breaks
        if translated_text_lt:
            translated_text_lt += "\n\n"
        translated_text_lt += translated_para

    return translated_text_lt.strip() # Return the final translated text


# --- Main Loop ---

# Pseudo-SQL for creating the necessary table structure.
# Adapt the data types (e.g., INT, VARCHAR, TEXT, NVARCHAR) for your specific database system.
"""
CREATE TABLE YOUR_SOURCE_TABLE_NAME (
    YOUR_ID_COLUMN INT PRIMARY KEY, -- Or another appropriate data type like BIGINT, UNIQUEIDENTIFIER
    YOUR_SOURCE_TEXT_COLUMN TEXT,   -- Or VARCHAR(MAX), NVARCHAR(MAX), etc. Stores the original English text.
    YOUR_TARGET_TEXT_COLUMN TEXT    -- Or VARCHAR(MAX), NVARCHAR(MAX), etc. Stores the translated Lithuanian text.
);

-- Optional: Add an index to speed up finding untranslated records
CREATE INDEX idx_untranslated ON YOUR_SOURCE_TABLE_NAME (YOUR_TARGET_TEXT_COLUMN)
WHERE YOUR_TARGET_TEXT_COLUMN IS NULL;
"""

def main_loop():
    """
    Main loop to continuously check for and process untranslated records.
    """
    print("Starting translation service...")
    while True:
        conn = None # Initialize connection variable
        try:
            print("\nChecking for new records to translate...")
            start_time = time.time() # Record start time for performance metrics

            # Connect to the database
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()

            # Fetch records where the target language column is NULL
            # Use placeholders for table and column names
            select_query = f"""
                SELECT TOP {DB_FETCH_BATCH_SIZE} {ID_COLUMN}, {SOURCE_TEXT_COLUMN}
                FROM {SOURCE_TABLE}
                WHERE {TARGET_TEXT_COLUMN} IS NULL
            """
            cursor.execute(select_query)
            rows = cursor.fetchall()

            # If no records are found, wait and poll again
            if not rows:
                print(f"No items to process. Waiting for 60 seconds...")
                cursor.close()
                conn.close()
                time.sleep(60) # Wait for 1 minute before checking again
                continue

            total_items_in_batch = len(rows)
            print(f"Found {total_items_in_batch} records to translate in this batch.")

            processed_items = 0
            update_batch_params = [] # List to store parameters for batch update

            # Process each fetched row
            for row in rows:
                item_id, text_to_translate = row

                # Perform the translation
                translated_description = translate_text_structure(text_to_translate)

                # Add the result to the batch update list
                update_batch_params.append((translated_description, item_id))
                processed_items += 1

                # Simple progress indicator within the batch
                print(f"\rProcessed {processed_items} / {total_items_in_batch}...", end="")

            # Update the database in a single batch after processing all fetched rows
            if update_batch_params:
                print(f"\nUpdating {len(update_batch_params)} records in the database...")
                # Use placeholders for table and column names
                update_query = f"""
                    UPDATE {SOURCE_TABLE}
                    SET {TARGET_TEXT_COLUMN} = ?
                    WHERE {ID_COLUMN} = ?
                """
                cursor.executemany(update_query, update_batch_params)
                conn.commit() # Commit the transaction

                end_time = time.time()
                elapsed_time = end_time - start_time
                avg_time_per_item = elapsed_time / processed_items if processed_items > 0 else 0
                print(f"Batch finished. Processed {processed_items} items in {elapsed_time:.2f} seconds "
                      f"(Avg: {avg_time_per_item:.2f} sec/item).")

            # Clean up database resources for this iteration
            cursor.close()
            conn.close()

        except pyodbc.Error as db_err:
            # Handle database-specific errors
            print(f"Database error occurred: {db_err}")
            print("Waiting for 10 seconds before retrying...")
            time.sleep(10)
        except Exception as e:
            # Handle other potential errors (e.g., model loading, translation issues)
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            print("Waiting for 10 seconds before retrying...")
            time.sleep(10) # Wait before trying again in case of an error
        finally:
            # Ensure the database connection is closed even if errors occur
            if conn:
                try:
                    conn.close()
                except pyodbc.Error:
                    pass # Ignore errors during close if connection is already problematic

# --- Script Execution ---

if __name__ == "__main__":
    # Entry point of the script
    main_loop()