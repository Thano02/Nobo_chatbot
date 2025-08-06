# app.py
import streamlit as st
from rag_chat_engine import ask_gpt_rag

st.set_page_config(
    page_title="Nobo House Camp Chatbot",
    page_icon="🔥",
    layout="centered"
)

# Logo / Titre
st.title("🏕️ Nobo House Camp — Assistant")
st.markdown("Have a question about the camp? (Schedule, meals, logistics, etc.)")

# Zone de saisie utilisateur
question = st.text_input("❓ What would you like to know?")

if question:
    with st.spinner("🧠 Bit of playa dust in the brain… but i'm thinking... 🤔"):
        try:
            response, sources = ask_gpt_rag(question)
            st.markdown("### 🤖 Answer :")
            st.success(response)

            with st.expander("📄 References :"):
                for src in sources:
                    st.markdown(f"- `{src}`")

        except Exception as e:
            st.error(f"❌ Erreur lors de la génération : {e}")
