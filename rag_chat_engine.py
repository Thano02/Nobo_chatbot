import os
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors

# 🔐 Clé OpenAI depuis variable d’environnement
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 📂 Récupérer le répertoire actuel du fichier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📦 Chargement des fichiers pkl avec chemins absolus
def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

all_texts = load_pickle("all_texts.pkl")
all_sources = load_pickle("all_sources.pkl")
embeddings = load_pickle("all_embeddings.pkl")

# 🔁 Rechargement du modèle d'embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🧠 Index sklearn
nn = NearestNeighbors(n_neighbors=6, metric='cosine')
nn.fit(embeddings)

# 🗨️ Fonction principale
def ask_gpt_rag(question, k=6):
    q_embedding = embedding_model.encode([question])
    distances, indices = nn.kneighbors(q_embedding, n_neighbors=k)

    chunks = [all_texts[i] for i in indices[0]]
    sources = set(all_sources[i] for i in indices[0])

    context = "\n\n".join(chunks)
    messages = [
        {"role": "system", "content": "Tu es un assistant utile pour le camp Nobo House. Réponds uniquement en t'appuyant sur le contexte fourni."},
        {"role": "user", "content": f"{context}\n\nQuestion : {question}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=700,
        temperature=0.3
    )

    return response.choices[0].message.content, sources
