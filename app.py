# app.py
import streamlit as st
from rag_chat_engine import ask_gpt_rag

st.set_page_config(
    page_title="Nobo House Camp Chatbot",
    page_icon="🔥",
    layout="centered"
)

# Logo / Titre
st.title("🏕️ Nobo House Camp — Assistant Doc")
st.markdown("Pose une question sur les infos du camp (planning, repas, logistique, etc.).")

# Zone de saisie utilisateur
question = st.text_input("❓ Que veux-tu savoir ?")

if question:
    with st.spinner("Réflexion en cours... 🤔"):
        try:
            response, sources = ask_gpt_rag(question)
            st.markdown("### 🤖 Réponse :")
            st.success(response)

            with st.expander("📄 Sources utilisées :"):
                for src in sources:
                    st.markdown(f"- `{src}`")

        except Exception as e:
            st.error(f"❌ Erreur lors de la génération : {e}")
