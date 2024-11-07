import pandas as pd
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

# Function to display hits as a DataFrame
def display_hits_as_df(response: VespaQueryResponse, fields) -> pd.DataFrame:
    records = []
    for hit in response.hits:
        record = {}
        for field in fields:
            record[field] = hit["fields"][field]
        records.append(record)
    return pd.DataFrame(records)


# Function for keyword search on Vespa
def keyword_search(app, search_query):
    query = {
        "yql": "select * from sources * where userQuery() limit 5",
        "query": search_query,
        "ranking": "bm25",
    }
    response = app.query(query)
    return display_hits_as_df(response, ["doc_id", "title"])


# Function for semantic search on Vespa (using embeddings)
def semantic_search(app, query):
    query = {
        "yql": "select * from sources * where ({targetHits:100}nearestNeighbor(embedding,e)) limit 5",
        "query": query,
        "ranking": "semantic",
        "input.query(e)": "embed(@query)"
    }
    response = app.query(query)
    return display_hits_as_df(response, ["doc_id", "title"])


# Function to process the Airbnb CSV to Vespa-compatible format and upload data to Vespa
def process_airbnb_csv_and_upload(input_file, app):
    """
    Processes an Airbnb listings CSV file and uploads it to a Vespa instance.
    
    Args:
      input_file (str): Path to the input CSV file containing Airbnb listings data.
      app (Vespa): Vespa application object to interact with Vespa.
    """
    # Load the Airbnb listings CSV
    listings = pd.read_csv(input_file)

    # Fill missing values
    required_columns = ['id', 'name', 'room_type', 'neighbourhood']
    listings[required_columns] = listings[required_columns].fillna('')

    # Create the "text" column combining relevant features
    listings["text"] = listings.apply(lambda row: f"Room type: {row['room_type']} Neighbourhood: {row['neighbourhood']}", axis=1)

    # Select and rename columns for Vespa
    listings = listings[['id', 'name', 'text']]
    listings.rename(columns={'name': 'title', 'id': 'doc_id'}, inplace=True)

    # Upload data to Vespa
    for _, row in listings.iterrows():
        document = {
            "fields": row.to_dict(),
            "id": f"doc::{row['doc_id']}"
        }
        app.index.put(document)
    
    print(f"Successfully uploaded {len(listings)} listings to Vespa.")


# Function to get embedding for a specific document
def get_embedding(doc_id, app):
    query = {
        "yql": f"select doc_id, title, text, embedding from content.doc where doc_id contains '{doc_id}'",
        "hits": 1
    }
    result = app.query(query)

    if result.hits:
        return result.hits[0]
    return None


# Initialize Vespa application (replace with your Vespa instance details)
app = Vespa(url="http://localhost", port=8080)

# Process Airbnb CSV and upload to Vespa
process_airbnb_csv_and_upload("sc_airbnb_listings.csv", app)

# Example query for keyword search
query = "spacious apartment"
df = keyword_search(app, query)
print(df.head())

# Example query for semantic search (embedding-based search)
query = "comfortable stay in the city"
df = semantic_search(app, query)
print(df.head())

# Get embedding of a specific document and perform semantic search using that embedding
embedding_doc = get_embedding("12345", app)
if embedding_doc:
    embedding_vector = embedding_doc["fields"]["embedding"]
    results = app.query({
        "yql": 'select * from content.doc where ({targetHits:5}nearestNeighbor(embedding, user_embedding))',
        "ranking.features.query(user_embedding)": str(embedding_vector),
        "ranking.profile": "recommendation"
    })
    df = display_hits_as_df(results, ["doc_id", "title", "text"])
    print(df.head())
