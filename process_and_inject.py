

import json
import pandas as pd
from embeddings import ChromaVectorStore
from sentence_transformers import SentenceTransformer
from config.settings import Config
import os

def process_and_inject_data_sentence_transformer(input_file_path: str, output_file_path: str):
    """
    Reads a JSON file, generates embeddings using the sentence-transformers library,
    adds them as a new column, saves the result, and injects the data into ChromaDB.

    Args:
        input_file_path (str): The path to the input JSON file (can be JSON or JSON Lines).
        output_file_path (str): The path to save the output JSON file with embeddings.
    """
    # --- 1. Load Input JSON into a DataFrame ---
    try:
        # Use lines=True to handle JSON Lines format (one JSON object per line)
        df = pd.read_json(input_file_path, lines=True)
        print(f"Successfully loaded {len(df)} documents from {input_file_path} into a DataFrame.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        print("Please create an 'input.json' file in the same directory with your data.")
        return
    except ValueError as e:
        print(f"Error reading JSON into DataFrame: {e}. Please check the file format.")
        return

    if 'page_content' not in df.columns or 'metadata' not in df.columns:
        print("Error: The input JSON must produce 'page_content' and 'metadata' columns.")
        return

    # --- 2. Generate Embeddings with sentence-transformers ---
    print(f"Initializing SentenceTransformer model from local path: {Config.EMBEDDING_MODEL}")
    # Load the model directly from the local path specified in the config
    model = SentenceTransformer(Config.EMBEDDING_MODEL)

    print("Generating embeddings for each document...")
    page_contents = df['page_content'].tolist()
    
    # For e5 models, it's crucial to add the "passage: " prefix to each document
    prefixed_contents = [f"passage: {text}" for text in page_contents]
    
    # Generate embeddings. The output is a numpy array.
    embeddings_np = model.encode(prefixed_contents, show_progress_bar=True)
    
    # Convert the numpy array to a list of lists for JSON serialization and injection
    embeddings = embeddings_np.tolist()

    # --- 3. Add Embeddings to DataFrame ---
    df['embedding'] = embeddings
    print("Embeddings generated and added to the DataFrame.")

    # --- 4. Save the Augmented DataFrame to JSON ---
    try:
        df.to_json(output_file_path, orient='records', indent=4)
        print(f"Augmented data with embeddings saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return

    # --- 5. Inject Data into ChromaDB ---
    print("\nInitializing ChromaDB for data injection...")
    vector_store = ChromaVectorStore()

    print("Extracting data for injection from DataFrame...")
    metadatas = df['metadata'].tolist()
    ids = [meta.get('id', f'doc_{i}') for i, meta in enumerate(metadatas)]

    print(f"Injecting {len(df)} documents into ChromaDB...")
    try:
        vector_store.add_documents_with_embeddings(
            page_contents=page_contents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        print("\nInjection complete!")
        print(f"Total documents in collection: {vector_store.get_collection_count()}")
    except Exception as e:
        print(f"An error occurred during ChromaDB injection: {e}")


if __name__ == "__main__":
    input_json_path = 'input.json'
    output_json_path = 'output.json'
    
    process_and_inject_data_sentence_transformer(input_json_path, output_json_path)
