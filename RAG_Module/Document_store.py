from sentence_transformers import SentenceTransformer
import chromadb

with open("emg_info.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Your file has {len(text)} characters.")
print()
print("Sample text:", text[:200])


def chunks_text(text, chunks_size=200, overlap=40):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunks_size])
        start += chunks_size - overlap
    return chunks


chunks = chunks_text(text)

model = SentenceTransformer("all-MiniLM-L6-V2")
embeddings = model.encode(chunks)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="emergency_docs")

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print(f"Total {collection.count()} documents stored successfully.")
print(f"{len(chunks)} chunks created -> shape of embeddings: {embeddings.shape}")


question = "What should I do during a fire emergency?"

q_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=q_embedding,
    n_results=1
)

retrieved = results["documents"][0]

print("\nRetrieved information:")
print(retrieved)