import os
import pickle
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors

# 🔐 Clé OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 🔁 Chargement des fichiers .pkl
def load_pickle(filename):
    full_path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Fichier introuvable : {full_path}")
    with open(full_path, "rb") as f:
        return pickle.load(f)

# 📂 Données pré-générées
all_texts = load_pickle("all_texts.pkl")
all_sources = load_pickle("all_sources.pkl")
embeddings = load_pickle("all_embeddings.pkl")

# 🤖 Index de similarité (Sklearn)
nn = NearestNeighbors(n_neighbors=6, metric="cosine")
nn.fit(embeddings)

# 📚 Fonction RAG principale
def ask_gpt_rag(question, k=6):
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Chargé uniquement à la demande

    # ⚠️ Calcul de l'embedding de la question (léger)
    q_embedding = embedding_model.encode([question])
    distances, indices = nn.kneighbors(q_embedding, n_neighbors=k)

    # 🔍 Récupération des passages similaires
    chunks = [all_texts[i] for i in indices[0]]
    sources = set(all_sources[i] for i in indices[0])

    context = "\n\n".join(chunks)
    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant utile pour le camp Nobo House. Réponds uniquement en t'appuyant sur le contexte fourni.",
        },
        {"role": "user", "content": f"{context}\n\nQuestion : {question}"}
    ]

    # 🧠 Requête OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=700,
        temperature=0.3,
    )

    return response.choices[0].message.content, sources
