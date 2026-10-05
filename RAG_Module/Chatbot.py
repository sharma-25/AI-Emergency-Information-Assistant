# My AI_Based Chatbot - "Emergency Information Assisstant" 
import chromadb
import streamlit as st
from ollama import chat
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Emergency Information Assistant",
    page_icon="🚨",
    layout="wide"
)

system_msg = """You are a friendly Emergency Information Assistant.
Answer only using the context provided.
If the answer is not in the context, say that you don't know.
Keep the answer short, clear and useful.
For life-threatening emergencies, advise the user to contact emergency services immediately.
"""


@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-V2")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="emergency_docs")
    return model, collection


def retrieve(question, k=3):
    q_emb = model.encode(question).tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return list(zip(
        res["ids"][0],
        res["documents"][0],
        res["distances"][0]
    ))


def to_similarity(dist):
    return max(0.0, 1 - dist / 2)


def badge(score):
    if score >= 0.5:
        return "🟢"
    if score >= 0.3:
        return "🟡"
    return "🔴"


model, collection = load_resources()

st.title("🚨 Emergency Information Assistant")
st.caption("Ask questions about emergency situations and safety information.")

with st.sidebar:
    st.header("🚨 Available Emergency Information")

    topics = [
        "🚑 Medical Emergency",
        "🩸 Heavy Bleeding",
        "🔥 Burns",
        "🔥 Fire Emergency",
        "🛢️ Gas Leak",
        "🚗 Road Accident",
        "⚡ Electrical Shock",
        "🌊 Drowning",
        "☠️ Poisoning",
        "🌪️ Natural Disaster",
        "🔎 Missing Person",
        "👮 Police Emergency",
        "🛡️ Women's Safety",
        "🧒 Child Emergency",
        "🚨 General Emergency Rule",
        "❓ Unknown Emergency"
    ]

    for topic in topics:
        st.write(topic)

    st.divider()

    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 5, 3)

st.session_state.setdefault("last_query", "-")
st.session_state.setdefault("results", [])
st.session_state.setdefault("answer", "")
st.session_state.setdefault("recent", [])

left, right = st.columns(2)

with left:
    st.subheader("Ask Emergency Assistant")

    question = st.text_input(
        "Type your emergency question:",
        placeholder="What should I do during a fire?"
    )

    search = st.button("Search")


if search and question:
    st.session_state.last_query = question
    st.session_state.results = retrieve(question, top_k)

    if question in st.session_state.recent:
        st.session_state.recent.remove(question)

    st.session_state.recent.insert(0, question)
    st.session_state.recent = st.session_state.recent[:5]


with right:
    st.subheader("Emergency Information")

    if not st.session_state.results:
        st.info("Retrieved emergency information will be displayed here.")

    for chunk_id, text, dist in st.session_state.results:
        score = to_similarity(dist)

        st.warning(
            f"**{chunk_id}** {badge(score)} Match: {score:.0%}\n\n{text}"
        )


with st.sidebar:
    st.divider()

    st.header("Knowledge Base")

    st.metric(
        "Total chunks in memory",
        collection.count()
    )

    st.caption("Last searched query")
    st.write(st.session_state.last_query)

    st.divider()

    st.subheader("Recent Searches")

    for q in st.session_state.recent:
        st.write(f"• {q}")


