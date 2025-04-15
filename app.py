import streamlit as st
from openai import OpenAI
from datetime import datetime
from typing import List

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Pathly - AI Career Coach", layout="wide")
st.title("🤖 Pathly - Your AI Career Coach")
st.markdown("Empowering you with emotionally intelligent, personalized career guidance ✨")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Pathly, an emotionally intelligent AI career coach. Your job is to deeply understand the user's emotional state, blockers, and life context, and then guide them with personalized, motivating, and structured roadmaps. You adapt your tone, routine, and roadmap intensity based on the user's feelings."}
    ]

# --- Sidebar: Emotional & Career Onboarding ---
st.sidebar.header("🧠 Emotion & Career Discovery")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=12, max_value=70, value=25)
background = st.sidebar.text_area("🧠 Education & Work Background")
interests = st.sidebar.text_area("🎯 Interests & Career Goals")
location = st.sidebar.text_input("🌍 Preferred Work Location")
motivation = st.sidebar.slider("🔥 Motivation Level", 0, 10, 7)

st.sidebar.markdown("---")
st.sidebar.subheader("❤️ How Are You Feeling Today?")
emotions = st.sidebar.multiselect("Select emotions that describe your current state:", [
    "Lost", "Anxious", "Burnt out", "Unmotivated", "Stuck", "Curious", "Hopeful", "Excited"
])
blocker = st.sidebar.text_area("📌 What's blocking your progress?")

# --- Mermaid Renderer ---
def render_mermaid_chart(mermaid_code):
    st.components.v1.html(
        f"""
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=600,
    )

# --- Emotionally Aware Prompt Builder ---
def build_emotion_aware_prompt():
    return f"""
    User: {name}, Age: {age}, Location: {location or 'Not specified'}

    Background: {background}
    Career Interests: {interests}
    Emotional State: {', '.join(emotions)}
    Motivation Level: {motivation}/10
    Blocker: {blocker}

    Please analyze the emotional patterns and generate:
    1. Emotional diagnosis and coaching tone
    2. Categorize problem (e.g., lack of clarity, burnout, fear of failure)
    3. Short/Mid/Long-term career roadmap with routines
    4. Suggest mood-based daily habit loops
    5. If relevant, add mermaid chart or Gantt chart
    """

# --- Trigger Roadmap Generation ---
if st.sidebar.button("🔄 Start My Journey"):
    if not (name and background and interests and blocker):
        st.warning("Please complete all fields including blockers and emotions.")
    else:
        with st.spinner("🧠 Interpreting your emotional and career context..."):
            prompt = build_emotion_aware_prompt()
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=st.session_state.messages,
                temperature=0.8
            )
            reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.success("Here’s your personalized career journey 💡")

# --- Chat Interface ---
st.subheader("💬 Ongoing Chat with Pathly")
user_query = st.chat_input("Ask about routine, mood tracker, or request a career diagram")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=st.session_state.messages,
        temperature=0.8
    )
    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Display Chat ---
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        if "```mermaid" in msg["content"]:
            parts = msg["content"].split("```mermaid")
            st.markdown(parts[0])
            mermaid_code = parts[1].strip().strip("`")
            render_mermaid_chart(mermaid_code)
        else:
            st.markdown(msg["content"])
