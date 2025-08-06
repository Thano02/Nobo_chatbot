import os
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors

# 🔐 Clé via variable d’environnement
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 📁 Dossier courant (emplacement de ce fichier)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 📦 Chemins des fichiers
TEXTS_PATH = os.path.join(CURRENT_DIR, "all_texts.pkl")
SOURCES_PATH = os.path.join(CURRENT_DIR, "all_sources.pkl")
EMBEDDINGS_PATH = os.path.join(CURRENT_DIR, "all_embeddings.pkl")

# 🔍 Chargement des fichiers
for path in [TEXTS_PATH, SOURCES_PATH, EMBEDDINGS_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")

with open(TEXTS_PATH, "rb") as f:
    all_texts = pickle.load(f)

with open(SOURCES_PATH, "rb") as f:
    all_sources = pickle.load(f)

with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)

# 🚀 Modèle d'embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔍 Index sklearn
nn = NearestNeighbors(n_neighbors=6, metric='cosine')
nn.fit(embeddings)

def ask_gpt_rag(question, k=6):
    q_embedding = embedding_model.encode([question])
    distances, indices = nn.kneighbors(q_embedding, n_neighbors=k)

    chunks = [all_texts[i] for i in indices[0]]
    sources = set(all_sources[i] for i in indices[0])

    context = "\n\n".join(chunks)
    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant utile pour le camp Nobo House. Réponds uniquement en t'appuyant sur le contexte fourni."
        },
        {
            "role": "user",
            "content": f"{context}\n\nQuestion : {question}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=700,
        temperature=0.3
    )

    return response.choices[0].message.content, sources
