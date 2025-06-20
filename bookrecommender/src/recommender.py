import time
import pandas as pd
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

# Constants
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "books_collection"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../data/books_cleaned.csv")


# Globals initialized once
embedding_model = None
collection = None
book_dataframe = None


def load_books(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    print("Loading Hugging Face sentence-transformer model...")
    start = time.time()
    model = SentenceTransformer(model_name)
    print(f"Model loaded in {time.time() - start:.2f} seconds.")
    return model


def initialize_chroma(persist_dir: str):
    print("Initializing Chroma client...")
    return chromadb.PersistentClient(path=persist_dir)


def create_or_load_collection(client, model, books: pd.DataFrame):
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print("Loading existing collection from local Chroma DB...")
        start = time.time()
        col = client.get_collection(name=COLLECTION_NAME)
        print(f"Collection loaded in {time.time() - start:.2f} seconds.")
        return col

    print("No existing collection found. Generating and persisting new search vectors...")

    descriptions = books["description"].fillna("").tolist()
    ids = books["isbn13"].astype(str).tolist()

    print("Generating embeddings...")
    start = time.time()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    print(f"Embeddings generated in {time.time() - start:.2f} seconds.")

    col = client.create_collection(name=COLLECTION_NAME)
    col.add(
        documents=descriptions,
        embeddings=embeddings,
        ids=ids,
        metadatas=books.to_dict(orient="records")
    )
    print("Embeddings persisted to Chroma DB.")
    return col


def retrieve_semantic_recommendations(query: str, top_k: int = 3) -> pd.DataFrame:
    print(f"Running semantic search for query: '{query}'")
    start = time.time()
    embedding = embedding_model.encode([query])[0]
    result = collection.query(query_embeddings=[embedding], n_results=top_k)
    print(f"Query completed in {time.time() - start:.3f} seconds.")
    matched_ids = [int(book_id) for book_id in result["ids"][0]]
    return book_dataframe[book_dataframe["isbn13"].isin(matched_ids)]


def main():
    global embedding_model, collection, book_dataframe

    book_dataframe = load_books(CSV_PATH)
    embedding_model = load_embedding_model()
    chroma_client = initialize_chroma(CHROMA_DIR)
    collection = create_or_load_collection(chroma_client, embedding_model, book_dataframe)

    # Sample test query
    result_df = retrieve_semantic_recommendations("Books about roman empire")
    cols = [c for c in ["title", "author", "description"] if c in result_df.columns]
    print(result_df[cols])


if __name__ == "__main__":
    main()
