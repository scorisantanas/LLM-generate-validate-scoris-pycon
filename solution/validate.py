# -*- coding: utf-8 -*-
"""
This script asynchronously fetches previously generated financial descriptions
from a database, validates them against the original financial data using a
local Large Language Model (LLM) via an API, and updates the validation status
('Correct' or 'Incorrect') back into the database table.

It uses asyncio and aiohttp for concurrent API calls and pyodbc for database interaction.
A semaphore limits the number of concurrent validation requests to the LLM API.
"""

# --- Standard Library Imports ---
import asyncio
import json
import random
import time
from datetime import datetime

# --- Third-party Library Imports ---
import aiohttp  # For asynchronous HTTP requests
import pyodbc  # For database connection (replace with your specific DB driver if needed)

# --- Configuration ---

# Database connection string placeholder.
# Replace with your actual database connection string.
# See the first script for examples (SQL Server, PostgreSQL).
conn_str = "YOUR_DATABASE_CONNECTION_STRING"

# LLM API endpoint configuration.
# Replace with the URL of your LLM API endpoint.
LLM_API_URL = "http://localhost:2242/v1/chat/completions" # Example for Aphrodite Engine

# LLM model identifier for validation.
# Replace with the specific model identifier you want to use for validation tasks.
LLM_MODEL_VALIDATION = "YOUR_LLM_MODEL_IDENTIFIER_FOR_VALIDATION" # e.g., "neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit"

# Target table name placeholder.
# This is the table containing the generated descriptions and where the validation status will be updated.
TARGET_TABLE = "your_target_table_name" # e.g., "llm_finance_descriptions"

# Concurrency control: Maximum number of simultaneous requests to the LLM API.
# Adjust based on your API server's capacity and rate limits.
CONCURRENT_REQUEST_LIMIT = 80

# --- Global Variables ---

# Semaphore to control the number of concurrent API calls.
semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

# --- Asynchronous Functions ---

async def generate_validation(session, prompt_text, retries=3, delay=2):
    """
    Sends a validation prompt to the LLM API and retrieves the validation label.

    Args:
        session (aiohttp.ClientSession): The HTTP client session.
        prompt_text (str): The complete prompt asking for validation.
        retries (int): Maximum number of retry attempts in case of API failure.
        delay (int): Delay in seconds between retries.

    Returns:
        str or None: The validation label ('Correct'/'Incorrect') from the LLM,
                     or None if the request fails or response is malformed.
    """
    headers = {"Content-Type": "application/json"}
    # Seed is commented out in the original, can be added if needed for reproducibility.
    # random_seed = random.randint(0, 1_000_000_000)

    # Structure the request payload for the validation task.
    # Modify this structure based on your LLM API requirements.
    data = {
        "model": LLM_MODEL_VALIDATION,
        "messages": [
            # System/Assistant messages can help prime the model, depending on the model.
            # {"role": "system", "content": "You validate facts in the text and label text 'CORRECT' or 'INCORRECT'"},
            # {"role": "assistant", "content": "Understood. I will validate facts..."},
            {"role": "user", "content": prompt_text}
        ],
        # Add or remove parameters based on API capabilities and desired behavior.
        # "mode": "instruct", # Specific to some LLM servers like Aphrodite
        # "seed": random_seed,
        "stream": False,
        "temperature": 0.5, # Lower temperature for more deterministic validation output
        "top_p": 1,
        "max_tokens": 10 # Limit response length, expecting just 'Correct' or 'Incorrect'
        # "stop": "<|eot_id|>" # Model-specific stop token if needed
    }

    # Attempt the API call with retries on failure.
    for attempt in range(retries):
        try:
            # Acquire semaphore before making the request.
            async with semaphore:
                 # Make the asynchronous POST request. ssl=False might be needed for local endpoints.
                async with session.post(LLM_API_URL, headers=headers, json=data, ssl=False) as response:
                    # Check for successful response status.
                    if response.status == 200:
                        response_json = await response.json()
                        # Safely extract the response content.
                        choices = response_json.get('choices')
                        if choices and len(choices) > 0:
                            message = choices[0].get('message')
                            if message:
                                content = message.get('content')
                                if content:
                                    # Return the stripped content (expected 'Correct' or 'Incorrect').
                                    return content.strip()
                        # Log unexpected response structure if content extraction fails.
                        print("\nError or unexpected response structure received:")
                        print(json.dumps(response_json, indent=2))
                        return None # Indicate failure or malformed response
                    else:
                        # Log error if response status is not OK.
                        error_message = await response.text()
                        print(f"\nRequest failed with status {response.status}: {error_message}. Retrying ({attempt+1}/{retries})...")

        except aiohttp.ClientError as e:
            # Handle client-side connection errors.
            print(f"\nRequest failed due to client error: {e}. Retrying ({attempt+1}/{retries})...")
        except Exception as e:
            # Handle other potential exceptions during the request.
            print(f"\nRequest failed due to an unexpected error: {e}. Retrying ({attempt+1}/{retries})...")

        # Wait before retrying.
        await asyncio.sleep(delay * (attempt + 1)) # Basic exponential backoff

    # If all retries fail, return None.
    print(f"\nFailed to get a validation response after {retries} attempts.")
    return None

# --- Helper Functions ---

def generate_prompt_text(row):
    """
    Constructs the prompt text for the LLM validation task based on row data.

    Args:
        row (pyodbc.Row): A row object from the database cursor, expected to have
                          attributes like financial_data, measurement_unit,
                          and generated_description_en.

    Returns:
        str: The fully formatted prompt string.
    """
    # Handle measurement unit display. Use 'n/a' or similar if not applicable.
    measurement_unit = row.measurement_unit if row.measurement_unit and row.measurement_unit.lower() != 'n\\a' else "N/A"

    # Prepare financial data string (assuming it's stored as JSON text).
    # Adapt cleaning/formatting as needed based on actual data format.
    try:
        # Attempt to load if it's a JSON string, then dump cleanly.
        data_for_prompt_str = json.dumps(json.loads(row.financial_data))
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON or None, use as plain text or default representation.
        data_for_prompt_str = str(row.financial_data) if row.financial_data is not None else "{}"
    # Basic cleaning example: remove quotes and backslashes if they interfere.
    clean_data = data_for_prompt_str.replace('"', '').replace('\\', '')

    # Construct the prompt for the LLM validation task.
    # Clearly state the instructions, provide the text to validate, and the data to validate against.
    prompt_text = (
        "You validate facts in the text and label text 'CORRECT' or 'INCORRECT'.\n"
        "Instruction:\n"
        "1. Validate the text against the provided data set.\n"
        "2. Respond with a single-word label: 'Correct' or 'Incorrect'.\n"
        "  - 'Correct' if the text generally matches or is approximately equal to the provided data (rounded values are acceptable).\n"
        "  - 'Incorrect' if the text contains factual inaccuracies compared to the provided data.\n\n"
        "# Text for Validation:\n"
        f"\"{row.generated_description_en}\"\n\n"
        "# Data for Validation:\n"
        f"\"{clean_data}\"\n\n"
        f"# Measurement Unit Context (if applicable):\n{measurement_unit}\n\n"
        f"Important: Pay attention to terms indicating trends (e.g., growth/decline) when comparing numbers over time.\n"
        f"Provide NO reasoning or explanation. Respond with the single word 'Correct' or 'Incorrect' only.\n\n"
        f"Your single-word assessment is:"
    )
    return prompt_text

def update_validation_in_db(cursor, row_id, validation_result):
    """
    Updates the validation status for a specific row in the database.

    Args:
        cursor (pyodbc.Cursor): The database cursor object.
        row_id (any): The unique identifier (ID) of the row to update.
        validation_result (str): The validation status ('Correct', 'Incorrect', or potentially None/Error).
    """
    # --- Pseudo CREATE TABLE Statement (Reference) ---
    # This script assumes a table structure like the following exists.
    # The 'validation' column is added or expected to exist for storing the result.
    #
    # CREATE TABLE your_target_table_name (
    #     ID INTEGER PRIMARY KEY, -- Or appropriate auto-incrementing type
    #     company_code INTEGER,
    #     prompt TEXT,
    #     seed INTEGER,
    #     model VARCHAR(255),
    #     financial_data TEXT,
    #     measurement_unit VARCHAR(50),
    #     generated_description_en TEXT,
    #     generated_description_lt TEXT,
    #     description_type VARCHAR(50),
    #     prompt_tokens INTEGER,
    #     completion_tokens INTEGER,
    #     total_tokens INTEGER,
    #     load_date_time TIMESTAMP,
    #     validation VARCHAR(20) -- << Column added/used by this script
    # );
    # ---------------------------------------------

    # Ensure validation_result is not None before updating.
    if validation_result is None:
        validation_result = "Error: No Response" # Or handle as needed

    # Define the SQL UPDATE statement using placeholders.
    # Adapt the table name, validation column name ('validation'), and ID column name ('ID') if needed.
    update_query = f"UPDATE {TARGET_TABLE} SET validation = ? WHERE ID = ?"
    try:
        # Execute the update command.
        cursor.execute(update_query, (validation_result, row_id))
        # Note: The commit is handled in the calling function (handle_row) in the original script's pattern.
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"\nDatabase error during update for ID {row_id} (SQLSTATE: {sqlstate}): {ex}")
    except Exception as e:
        print(f"\nUnexpected error during database update for ID {row_id}: {e}")


def print_status(current_item_index, total_items, row_start_time, row_end_time, script_start_time):
    """
    Calculates and prints the processing status, including progress percentage and estimated time remaining.

    Args:
        current_item_index (int): The index (1-based) of the item just processed.
        total_items (int): The total number of items to process.
        row_start_time (float): Timestamp when processing the current item started.
        row_end_time (float): Timestamp when processing the current item ended.
        script_start_time (float): Timestamp when the main script execution started.
    """
    # Calculate time taken for the current item.
    item_execution_time = row_end_time - row_start_time
    # Calculate total time elapsed since the script started.
    total_elapsed_time = row_end_time - script_start_time
    # Calculate average time per item processed so far.
    avg_time_per_item = total_elapsed_time / current_item_index if current_item_index > 0 else 0
    # Calculate completion percentage.
    percent_complete = (current_item_index / total_items) * 100 if total_items > 0 else 0
    # Estimate remaining time based on average time per item.
    remaining_items = total_items - current_item_index
    estimated_remaining_sec = avg_time_per_item * remaining_items if avg_time_per_item > 0 else 0
    estimated_remaining_hr = round(estimated_remaining_sec / 3600, 1)

    # Format times for display.
    rounded_item_time = round(item_execution_time, 1)
    avg_rounded_time = round(avg_time_per_item, 1)

    # Print the status update to the console, overwriting the previous line.
    print(f'\rProcessing {current_item_index}/{total_items}. {percent_complete:.2f}% complete. '
          f'Last item: {rounded_item_time}s. Avg time/item: {avg_rounded_time}s. '
          f'Est. remaining: {estimated_remaining_hr} hrs.   ', end='')

# --- Core Processing Logic ---

async def handle_row(session, row, cursor, item_index, total_rows, script_start_time):
    """
    Handles the processing for a single row: generate prompt, get validation, update DB.

    Args:
        session (aiohttp.ClientSession): The shared HTTP client session.
        row (pyodbc.Row): The database row containing data for validation.
        cursor (pyodbc.Cursor): The database cursor for updates.
        item_index (int): The 1-based index of the current row being processed.
        total_rows (int): The total number of rows to process.
        script_start_time (float): The timestamp when the overall script started.
    """
    row_start_time = time.time()
    try:
        # Generate the specific prompt for this row's data.
        prompt_text = generate_prompt_text(row)
        # Call the LLM API to get the validation result ('Correct'/'Incorrect').
        validation_result = await generate_validation(session, prompt_text)

        # Update the database with the validation result for the current row's ID.
        # Assumes the row object has an 'ID' attribute corresponding to the primary key.
        update_validation_in_db(cursor, row.ID, validation_result)
        # Commit the transaction after each row update, as per the original script's pattern.
        # Note: Batching commits might be more efficient for large datasets.
        cursor.connection.commit()

    except Exception as e:
        # Log any unexpected errors during row processing.
        print(f"\nError handling row ID {row.ID if row else 'unknown'}: {e}")
        # Optionally rollback if transaction integrity is critical: cursor.connection.rollback()

    finally:
        # Record end time and print status update regardless of success or failure.
        row_end_time = time.time()
        print_status(item_index, total_rows, row_start_time, row_end_time, script_start_time)


async def process_data(cursor):
    """
    Fetches data rows needing validation and orchestrates their asynchronous processing.

    Args:
        cursor (pyodbc.Cursor): The database cursor for fetching data.
    """
    script_start_time = time.time() # Record start time for overall progress

    # --- Fetch Data from Target Database ---
    # Define the SQL query to fetch rows that haven't been validated yet.
    # Selects necessary columns: ID, financial_data, measurement_unit, generated_description_en.
    # Adapt table and column names as needed. The WHERE clause filters for NULL validation status.
    query = f"""
        SELECT
            ID, financial_data, measurement_unit, generated_description_en
        FROM {TARGET_TABLE}
        WHERE validation IS NULL -- Fetch rows where validation hasn't been performed
        -- Add other conditions if needed, e.g.: AND description_type = 'finance'
    """
    print(f"Executing SQL query to fetch data needing validation from {TARGET_TABLE}...")
    cursor.execute(query)
    rows = cursor.fetchall() # Fetch all relevant rows into memory
    total_rows = len(rows)
    print(f"Done. Fetched {total_rows} rows to validate.")

    if not rows:
        print("No rows found needing validation.")
        return

    # --- Process Rows Asynchronously ---
    print(f"Starting asynchronous validation processing with concurrency limit {CONCURRENT_REQUEST_LIMIT}...")
    # Create a persistent HTTP session for reuse across API calls.
    async with aiohttp.ClientSession() as session:
        tasks = [] # List to hold asynchronous tasks
        # Iterate through fetched rows, creating a task for each.
        for i, row in enumerate(rows, start=1):
            # Create and schedule the handle_row coroutine as a task.
            task = asyncio.create_task(handle_row(session, row, cursor, i, total_rows, script_start_time))
            tasks.append(task)

            # Optional: Gather tasks periodically if memory usage is a concern for very large datasets.
            # This variant gathers tasks after every CONCURRENT_REQUEST_LIMIT are created.
            # if len(tasks) >= CONCURRENT_REQUEST_LIMIT:
            #     await asyncio.gather(*tasks) # Wait for the current batch of tasks
            #     tasks = [] # Reset tasks list

        # Wait for any remaining tasks to complete after the loop.
        if tasks:
            await asyncio.gather(*tasks)

    # Print a final newline after progress updates are finished.
    print("\nValidation processing finished.")
    total_duration = time.time() - script_start_time
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours).")

# --- Main Execution Block ---

async def main():
    """
    Main asynchronous function to set up the database connection and run the processing.
    """
    conn = None
    cursor = None
    print("Connecting to the database...")
    try:
        # Establish database connection.
        conn = pyodbc.connect(conn_str, autocommit=False) # Control commits manually
        cursor = conn.cursor()
        # Start the data processing workflow.
        await process_data(cursor)
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"Database connection or execution error (SQLSTATE: {sqlstate}): {ex}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure database resources are closed properly.
        if cursor:
            cursor.close()
            print("Database cursor closed.")
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    # Run the main asynchronous event loop.
    asyncio.run(main())