import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import os
from pathlib import Path

import streamlit as st
from app import Orchestrator

load_dotenv()  # Loads variables from .env

def _save_chat_history(name: str, phone: str, messages: list):
    os.makedirs("chat_history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history/chat_{timestamp}.txt"
    lines = [
        f"Name: {name or '(not provided)'}",
        f"Phone: {phone or '(not provided)'}",
        "",
        "--- Chat History ---",
        "",
    ]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


col1, col2 = st.columns([1, 6])
with col1:
    BASE_DIR = Path(__file__).parent
    logo_path = BASE_DIR / "logo_recruiting_agent.jpeg"
    st.image(str(logo_path), width=120)
with col2:
    st.title("Recruiting Agent")

st.markdown("<br></br>", unsafe_allow_html=True)

# Sidebar — candidate info
st.sidebar.title("👤 Candidate Info")
candidate_name = st.sidebar.text_input("Full Name")
candidate_phone = st.sidebar.text_input("Phone Number")

st.subheader("Tell us which role you find interesting and if there is any additional info you would like to know")
st.write("")
st.markdown(
    '<p style="font-size:25px">You can ask questions about the role itself and, if you meet the criteria, ask to schedule an interview.</p>',
    unsafe_allow_html=True,
)

# Per-session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

# Render full conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Show banner if chat has ended
if st.session_state.chat_ended:
    st.info("This conversation has ended. Refresh the page to start a new one.")

# Chat input (disabled after exit)
prompt = st.chat_input("Your prompt", disabled=st.session_state.chat_ended)

if prompt and not st.session_state.chat_ended:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    result = st.session_state.orchestrator.orchestrate_conversation_with_memory(
                        prompt, session_id=st.session_state.session_id)
    response = result["response"]

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if result.get("label") == 'end':
        st.session_state.chat_ended = True
        _save_chat_history(
            name=candidate_name,
            phone=candidate_phone,
            messages=st.session_state.messages,
        )
        st.rerun()
