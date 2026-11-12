import streamlit as st
import requests

# Konfiguration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/chat"

# Titel
st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your knowledge base.")

# Chatverlauf initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []

# Seitenleiste
with st.sidebar:
    st.header("RAG Chatbot")
    st.write("Powered by FastAPI, Supabase and Groq")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Bisherige Nachrichten anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.json(source)

# Texteingabe
user_input = st.chat_input("Ask me anything about the documents...")

if user_input:

    # Benutzernachricht speichern
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Antwort vom Backend abrufen
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"message": user_input},
                    timeout=60
                )

                response.raise_for_status()
                data = response.json()

                answer = data["response"]
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.json(source)

                # Antwort speichern
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except requests.RequestException as error:
                st.error("Could not connect to the chatbot backend.")
                st.caption(str(error))
