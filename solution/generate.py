# -*- coding: utf-8 -*-
"""
This script asynchronously fetches financial data for companies from a database,
generates financial descriptions using a local Large Language Model (LLM)
via an API, and inserts the results back into a database table.

It uses asyncio and aiohttp for concurrent API calls and pyodbc for database interaction.
A semaphore is used to limit the number of concurrent requests to the LLM API.
"""

# --- Standard Library Imports ---
import asyncio
import json
import random
import time
from datetime import datetime

# --- Third-party Library Imports ---
import aiohttp  # For asynchronous HTTP requests
import pandas as pd  # For data manipulation (specifically, accumulating results)
import pyodbc  # For database connection (replace with your specific DB driver if needed)

# --- Configuration ---

# Database connection string placeholder.
# Replace with your actual database connection string.
# Example for SQL Server:
# conn_str = (
#     r'DRIVER={ODBC Driver 17 for SQL Server};'
#     r'SERVER=your_server_name;'
#     r'DATABASE=your_database_name;'
#     r'UID=your_username;'
#     r'PWD=your_password;'
# )
# Example for PostgreSQL (using psycopg2 driver via ODBC):
# conn_str = (
#     r'DRIVER={PostgreSQL Unicode};'
#     r'SERVER=your_server_ip_or_hostname;'
#     r'PORT=5432;'
#     r'DATABASE=your_database_name;'
#     r'UID=your_username;'
#     r'PWD=your_password;'
# )
conn_str = "YOUR_DATABASE_CONNECTION_STRING"

# LLM API endpoint configuration.
# Replace with the URL of your LLM API endpoint.
LLM_API_URL = "http://localhost:2242/v1/chat/completions" # Example for Aphrodite Engine

# LLM model identifier.
# Replace with the specific model identifier you want to use, matching the model loaded in your LLM API server.
LLM_MODEL = "YOUR_LLM_MODEL_IDENTIFIER" # e.g., "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

# Source and target table names placeholders.
# Replace with your actual table names.
SOURCE_TABLE = "your_source_table_name" # Table containing company_code and financial_data
TARGET_TABLE = "your_target_table_name" # Table where generated descriptions will be stored

# Concurrency control: Maximum number of simultaneous requests to the LLM API.
# Adjust based on your API server's capacity and rate limits.
CONCURRENT_REQUEST_LIMIT = 30

# Batch size for database insertion.
# Results are accumulated and inserted into the database in batches of this size.
DB_BATCH_SIZE = 30

# --- Global Variables ---

# Semaphore to control the number of concurrent API calls.
semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

# Define the structure (columns) for the results DataFrame.
# This DataFrame temporarily holds results before batch insertion into the database.
df_columns = [
    'company_code', 'prompt', 'seed', 'model', 'financial_data',
    'measurement_unit', 'generated_description_en', 'generated_description_lt',
    'description_type', 'prompt_tokens', 'completion_tokens', 'total_tokens',
    'load_date_time'
]
# Initialize an empty DataFrame to store results.
# This DataFrame is cleared after each batch insertion.
results_df = pd.DataFrame(columns=df_columns)

# --- Asynchronous Functions ---

async def generate_description(session, prompt_text, retries=3, delay=2):
    """
    Sends a prompt to the LLM API and retrieves the generated text.

    Args:
        session (aiohttp.ClientSession): The HTTP client session.
        prompt_text (str): The complete prompt to send to the LLM.
        retries (int): Maximum number of retry attempts in case of API failure.
        delay (int): Delay in seconds between retries.

    Returns:
        tuple: A tuple containing:
            - str: The generated description text.
            - dict: The full JSON response from the API.
            - int: The random seed used for generation.

    Raises:
        Exception: If the API request fails after all retries.
    """
    headers = {"Content-Type": "application/json"}
    # Use a random seed for potentially varied but reproducible results if needed later.
    random_seed = random.randint(0, 1_000_000_000)

    # Structure the request payload according to the expected API format.
    # Modify this structure if your LLM API requires a different format.
    data = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional business writer, which strictly follows instructions"
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        # Add or remove parameters based on your API's capabilities (e.g., temperature, max_tokens)
        # "mode": "instruct", # Specific to some LLM servers like Aphrodite
        "seed": random_seed,
        "stream": False # Assuming non-streaming response is sufficient
    }

    # Attempt the API call with retries on failure.
    for attempt in range(retries):
        try:
            # Make the asynchronous POST request. ssl=False might be needed for local endpoints without valid certs.
            async with session.post(LLM_API_URL, headers=headers, json=data, ssl=False) as response:
                # Check for successful response status.
                if response.status == 200:
                    response_json = await response.json()
                    # Extract the relevant parts of the response. Adjust accessors based on actual API response structure.
                    generated_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                    return generated_text, response_json, random_seed
                else:
                    # Log error if response status is not OK.
                    error_message = await response.text()
                    print(f"Request failed with status {response.status}: {error_message}. Retrying ({attempt+1}/{retries})...")

        except aiohttp.ClientError as e:
            # Handle client-side connection errors.
            print(f"Request failed due to client error: {e}. Retrying ({attempt+1}/{retries})...")
        except Exception as e:
            # Handle other potential exceptions during the request.
            print(f"Request failed due to an unexpected error: {e}. Retrying ({attempt+1}/{retries})...")

        # Wait before retrying.
        await asyncio.sleep(delay * (attempt + 1)) # Exponential backoff could also be implemented here

    # If all retries fail, raise an exception.
    raise Exception(f"Failed to get a response from LLM API after {retries} attempts.")

async def insert_results_to_db(df_batch, start_time, total_rows_to_process, items_processed_so_far):
    """
    Inserts a batch of results from a DataFrame into the target database table.
    Also calculates and prints the progress.

    Args:
        df_batch (pd.DataFrame): DataFrame containing the results to insert.
        start_time (float): The timestamp when the main script started.
        total_rows_to_process (int): The total number of rows to be processed.
        items_processed_so_far (int): The number of items processed up to this batch.
    """
    if df_batch.empty:
        print("No data to insert.")
        return

    # --- Pseudo CREATE TABLE Statement ---
    # Ensure you have a table in your database that matches this structure.
    # Adapt the data types based on your specific database system (e.g., VARCHAR, TEXT, INT, FLOAT, TIMESTAMP).
    #
    # CREATE TABLE your_target_table_name (
    #     company_code INTEGER, -- Or appropriate numeric type for company identifier
    #     prompt TEXT, -- Or VARCHAR(MAX) / CLOB, stores the prompt sent to the LLM
    #     seed INTEGER, -- Stores the random seed used for generation
    #     model VARCHAR(255), -- Stores the identifier of the LLM used
    #     financial_data TEXT, -- Or VARCHAR(MAX) / CLOB, stores the input financial data JSON/text
    #     measurement_unit VARCHAR(50), -- Unit of measurement if applicable (e.g., 'EUR', 'thousands')
    #     generated_description_en TEXT, -- Or VARCHAR(MAX) / CLOB, the generated description in English
    #     generated_description_lt TEXT, -- Placeholder for potential translation (or remove if not needed)
    #     description_type VARCHAR(50), -- Type of description generated (e.g., 'finance')
    #     prompt_tokens INTEGER, -- Number of tokens in the prompt
    #     completion_tokens INTEGER, -- Number of tokens in the generated completion
    #     total_tokens INTEGER, -- Total tokens used for the API call
    #     load_date_time TIMESTAMP -- Timestamp when the record was loaded
    #     -- Add primary keys, indexes, or constraints as needed
    #     -- PRIMARY KEY (company_code, description_type) -- Example composite key
    # );
    # ------------------------------------

    # Define the SQL INSERT statement. Ensure the number of placeholders (?) matches the number of columns.
    # Adapt the table name and column names if they differ in your database.
    insert_query = f"""
    INSERT INTO {TARGET_TABLE} (
        company_code, prompt, seed, model, financial_data, measurement_unit,
        generated_description_en, generated_description_lt, description_type,
        prompt_tokens, completion_tokens, total_tokens, load_date_time
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """ # Using '?' as placeholder, adapt if your DB driver uses %s or other markers

    # Convert DataFrame rows to a list of tuples for bulk insertion.
    records_to_insert = list(df_batch.itertuples(index=False, name=None))

    conn = None
    cursor = None
    try:
        # Establish database connection.
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        # Execute the insertion command for multiple rows.
        # `executemany` is generally more efficient than inserting row by row.
        cursor.executemany(insert_query, records_to_insert)
        # Commit the transaction to save changes.
        conn.commit()

        # Calculate and print progress statistics.
        elapsed_time = time.time() - start_time
        avg_time_per_item = elapsed_time / items_processed_so_far if items_processed_so_far > 0 else 0
        percentage_complete = (items_processed_so_far / total_rows_to_process) * 100 if total_rows_to_process > 0 else 0
        estimated_total_time = avg_time_per_item * total_rows_to_process if avg_time_per_item > 0 else 0
        estimated_remaining_time_sec = estimated_total_time - elapsed_time
        estimated_remaining_time_hr = round(estimated_remaining_time_sec / 3600, 1) if estimated_remaining_time_sec > 0 else 0

        print(f'\rBatch inserted {len(df_batch)} records. Total processed: {items_processed_so_far}/{total_rows_to_process}. '
              f'{percentage_complete:.2f}% complete. Avg time/item: {avg_time_per_item:.2f}s. '
              f'Est. remaining: {estimated_remaining_time_hr} hrs.   ', end='')

    except pyodbc.Error as ex:
        # Handle potential database errors (e.g., connection issues, SQL errors).
        sqlstate = ex.args[0]
        print(f"\nDatabase error occurred (SQLSTATE: {sqlstate}): {ex}")
        # Consider adding rollback logic here if needed: conn.rollback()
    except Exception as e:
        # Handle other unexpected errors during database operation.
        print(f"\nAn unexpected error occurred during database insertion: {e}")
    finally:
        # Ensure database resources are closed properly.
        if cursor:
            cursor.close()
        if conn:
            conn.close()


async def process_company_row(session, row_data, semaphore_lock):
    """
    Processes a single row of company data: prepares the prompt, calls the LLM API,
    and appends the result to the global results DataFrame.

    Args:
        session (aiohttp.ClientSession): The HTTP client session.
        row_data (tuple): A tuple representing a row fetched from the source database table.
                          Expected format: (company_code, financial_data_json_or_text)
        semaphore_lock (asyncio.Semaphore): The semaphore to control concurrency.
    """
    global results_df # Declare intent to modify the global DataFrame

    async with semaphore_lock: # Acquire the semaphore before proceeding
        try:
            # Extract data from the row. Adapt indices if your source query returns columns differently.
            company_code = row_data[0] # Assuming company_code is the first column
            financial_data = row_data[1] # Assuming financial_data (JSON/text) is the second column

            # Prepare the financial data for the prompt (e.g., clean JSON string).
            # This assumes financial_data is fetched as a JSON string. Adapt if it's structured differently.
            try:
                # Attempt to load if it's a JSON string, then dump cleanly.
                # If it's already plain text, this might fail, handle accordingly.
                data_for_prompt_str = json.dumps(json.loads(financial_data))
            except json.JSONDecodeError:
                # If it's not valid JSON, use it as plain text (or apply other cleaning).
                data_for_prompt_str = str(financial_data)
            except TypeError:
                 # Handle cases where financial_data might be None or not string-like
                data_for_prompt_str = str(financial_data) if financial_data is not None else "{}"


            # Remove potentially problematic characters for the LLM prompt, if necessary.
            clean_data = data_for_prompt_str.replace('"', '').replace('\\', '') # Basic cleaning example

            # Construct the prompt for the LLM.
            # Clearly define the task, input data, and constraints/requirements.
            prompt_text = (
                f"Generate a detailed financial description for the company using the provided data only.\n"
                f"Important! You must use this data only:\n{clean_data}\n"
                f"Requirements:\n"
                f"1. Aim for 300-400 words description.\n"
                f"2. Ensure all figures are correct.\n"
                f"3. Use simple sentences and words.\n"
                f"4. Refer to the entity as 'the Company.'\n"
                f"5. Do not include any notes, explanations, or assumptions.\n"
                f"6. Do not include summary facts or footnotes.\n"
                f"7. It should be ready for publishing as-is copy-paste, so do not use any placeholders, introductions etc (e.g. do not use 'Sure, here it is..', 'Here is the financial description','Based on above requirements' and similar intro phrases). Just start with company description as requested.\n"
                f"8. Description must be SEO friendly with main keywords: financial data, financial reports, revenue, profit, finance\n"
            )

            # Call the LLM API to generate the description.
            generated_description_en, response_json, random_seed = await generate_description(session, prompt_text)

            # Extract metadata from the API response. Adjust accessors based on actual API response structure.
            model = response_json.get('model', LLM_MODEL) # Use configured model as fallback
            usage = response_json.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

            # Get the current timestamp for the load record.
            current_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Prepare the new row as a dictionary.
            new_row = {
                'company_code': company_code,
                'prompt': prompt_text,
                'seed': random_seed,
                'model': model,
                'financial_data': financial_data, # Store original input data
                'measurement_unit': 'n/a', # Set as applicable, or fetch from data
                'generated_description_en': generated_description_en,
                'generated_description_lt': None, # Placeholder for translation
                'description_type': 'finance', # Type identifier for this generation task
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'load_date_time': current_dt
            }

            # Append the result row to the global DataFrame.
            # Using _append is deprecated in newer pandas; consider concat or list accumulation if performance critical.
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        except Exception as e:
            # Log errors encountered during processing a single row.
            print(f"\nError processing row for company code {row_data[0] if row_data else 'unknown'}: {e}")
            # Optionally, add failed rows to a separate list/log for later review.

# --- Main Execution Block ---

async def main():
    """
    Main function to orchestrate the data fetching, processing, and insertion workflow.
    """
    global results_df # Declare intent to use and potentially modify the global DataFrame

    items_processed = 0
    total_rows = 0
    rows_to_process = []

    # --- Fetch Data from Source Database ---
    print("Connecting to the database to fetch source data...")
    conn = None
    cursor = None
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Define the SQL query to fetch data.
        # Selects companies from the source table that do not yet have a 'finance' description
        # in the target table. Fetches data in random order using 'newid()' (SQL Server specific)
        # or 'RANDOM()' (PostgreSQL/SQLite). Adapt ordering for your DB if needed.
        # Ensure column names (company_code, financial_data) and table names match your schema.
        query = f"""
        SELECT
            f.company_code,
            f.financial_data
        FROM {SOURCE_TABLE} f
        WHERE NOT EXISTS (
            SELECT 1
            FROM {TARGET_TABLE} ff
            WHERE ff.company_code = f.company_code
            AND ff.description_type = 'finance' -- Filter based on the specific description type
        )
        ORDER BY NEWID(); -- Or RANDOM() for PostgreSQL/SQLite, or remove for default order
        """ # Adapt ORDER BY clause for your database system if needed

        print(f"Executing SQL query to fetch data from {SOURCE_TABLE}...")
        cursor.execute(query)
        rows_to_process = cursor.fetchall() # Fetch all rows into memory
        total_rows = len(rows_to_process)
        print(f"Done. Fetched {total_rows} rows to process.")

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"Database error occurred during data fetching (SQLSTATE: {sqlstate}): {ex}")
        return # Exit if data fetching fails
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")
        return # Exit on other errors
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Database connection closed.")

    if not rows_to_process:
        print("No new data to process.")
        return

    # --- Process Data Asynchronously ---
    print("Starting asynchronous processing of rows...")
    start_time = time.time() # Record start time for progress calculation

    # Create an asynchronous HTTP session that will be reused for all API calls.
    async with aiohttp.ClientSession() as session:
        tasks = [] # List to hold concurrent tasks

        # Iterate through the fetched rows.
        for i, row in enumerate(rows_to_process):
            # Create a task for processing each row concurrently.
            task = asyncio.create_task(process_company_row(session, row, semaphore))
            tasks.append(task)

            # Check if the batch is full or if it's the last row.
            # The batch logic processes DB_BATCH_SIZE LLM API calls concurrently,
            # then waits for them to finish, then inserts the accumulated results.
            if (i + 1) % DB_BATCH_SIZE == 0 or (i + 1) == total_rows:
                # Wait for all tasks in the current batch to complete.
                await asyncio.gather(*tasks)

                # Once the tasks (API calls and DataFrame appends) are done,
                # insert the accumulated results into the database.
                if not results_df.empty:
                    current_batch_size = len(results_df)
                    # Update total processed count *before* calling insert for accurate reporting.
                    items_processed += current_batch_size
                    # Perform the batch database insertion.
                    await insert_results_to_db(results_df, start_time, total_rows, items_processed)
                    # Reset the global DataFrame for the next batch.
                    results_df = pd.DataFrame(columns=df_columns) # Re-initialize empty DataFrame

                # Reset the tasks list for the next batch.
                tasks = []

    # Final progress message newline after the loop finishes.
    print("\nProcessing finished.")
    total_duration = time.time() - start_time
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours).")

if __name__ == "__main__":
    # Run the main asynchronous function.
    asyncio.run(main())