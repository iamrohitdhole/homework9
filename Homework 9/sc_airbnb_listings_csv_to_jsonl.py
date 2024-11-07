import pandas as pd
import json


def collapse_airbnb_features(row):
    """
    Combines specific features into a single text field.
    """
    features = [
        f"Room type: {row['room_type']}", 
        f"Neighbourhood: {row['neighbourhood']}"
    ]
    return " ".join(features)


def process_airbnb_csv(input_file, output_file):
    """
    Processes an Airbnb listings CSV file to create a Vespa-compatible JSON format.

    This function reads a CSV file containing Airbnb listings data, processes the data to
    generate new columns for text search, and outputs a JSON file with the necessary
    fields (`put` and `fields`) for indexing documents in Vespa.

    Args:
      input_file (str): The path to the input CSV file containing Airbnb listings data.
                        Expected columns are 'id', 'name', 'room_type', and 'neighbourhood'.
      output_file (str): The path to the output JSON file to save the processed data in
                         Vespa-compatible format.

    Workflow:
      1. Reads the CSV file into a Pandas DataFrame.
      2. Fills missing values in the 'name', 'room_type', and 'neighbourhood' columns with empty strings.
      3. Creates a "text" column that combines specified features using the `collapse_airbnb_features` function.
      4. Selects and renames columns to match required Vespa format: 'doc_id', 'title', and 'text'.
      5. Constructs a JSON-like 'fields' column that includes the record's data.
      6. Creates a 'put' column based on 'doc_id' to uniquely identify each document.
      7. Outputs the processed data to a JSON file in a Vespa-compatible format.

    Returns:
      None. Writes the processed DataFrame to `output_file` as a JSON file.

    Notes:
      - The function requires the helper function `collapse_airbnb_features` to be defined, 
        which combines text features for the "text" column.
      - Output JSON file is saved with `orient='records'` and `lines=True` to create line-delimited JSON.

    Example Usage:
      >>> process_airbnb_csv("sc_airbnb_listings.csv", "output_vespa.json")
    """
    # Load the Airbnb listings CSV
    listings = pd.read_csv(input_file)
    
    # Check for missing required columns
    required_columns = ['id', 'name', 'room_type', 'neighbourhood']
    for col in required_columns:
        if col not in listings.columns:
            raise ValueError(f"Missing required column: {col}")

    # Fill missing values in required columns with empty strings or default values
    listings[required_columns] = listings[required_columns].fillna('')
    
    # Create a "text" column that combines specific features using the `collapse_airbnb_features` function
    listings["text"] = listings.apply(collapse_airbnb_features, axis=1)

    # Select only 'id', 'name', and 'text' columns for Vespa
    listings = listings[['id', 'name', 'text']]
    listings.rename(columns={'name': 'title', 'id': 'doc_id'}, inplace=True)

    # Create 'fields' column as JSON-like structure of each record
    listings['fields'] = listings.apply(lambda row: row.to_dict(), axis=1)

    # Create 'put' column based on 'doc_id'
    listings['put'] = listings['doc_id'].apply(lambda x: f"id:hybrid-search:doc::{x}")

    # Select the columns required for Vespa-compatible JSON output
    df_result = listings[['put', 'fields']]
    
    # Print a sample of the resulting DataFrame to verify structure
    print(df_result.head())
    
    # Write to JSON file in Vespa-compatible format
    df_result.to_json(output_file, orient='records', lines=True)

# Run the processing function on the Airbnb data
process_airbnb_csv("sc_airbnb_listings.csv", "clean_airbnb_listings.jsonl")
