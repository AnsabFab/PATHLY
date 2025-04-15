import streamlit as st
from openai import OpenAI
import os
from datetime import datetime
from typing import List

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Pathly - AI Career Coach", layout="wide")
st.title("🤖 Pathly - Your AI Career Coach")
st.markdown("Empowering you with smart, personalized, and visual career advice ✨")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Pathly, an expert AI career coach, capable of creating detailed strategies, skill-gap analyses, motivational insights, and visual diagrams like mermaid charts, Gantt charts, and weekly routines."}
    ]

# --- Sidebar Input ---
st.sidebar.header("📌 Tell us about yourself")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=12, max_value=70, value=25)
background = st.sidebar.text_area("🧠 Education & Work Background")
interests = st.sidebar.text_area("🎯 Interests & Career Goals")
location = st.sidebar.text_input("🌍 Preferred Work Location (Optional)")
motivation = st.sidebar.slider("🔥 Motivation Level", 0, 10, 7)

# --- Mermaid Renderer ---
def render_mermaid_chart(mermaid_code):
    st.components.v1.html(
        f"""
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({ startOnLoad: true });
        </script>
        """,
        height=600,
    )

# --- Agent Logic (Initial Prompt) ---
def generate_initial_plan():
    initial_prompt = f"""
    The user is {name}, aged {age}, located in {location or 'anywhere'}, with the following background:
    {background}

    Their career goals and interests are: {interests}.
    Motivation level: {motivation}/10.

    Please provide:
    1. A career strategy with steps and timelines
    2. Skill gap analysis and recommendations
    3. Job market insight (current roles + trends)
    4. Motivational guidance
    5. If applicable, a mermaid chart or Gantt chart
    """

    st.session_state.messages.append({"role": "user", "content": initial_prompt})
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=st.session_state.messages,
        temperature=0.75
    )
    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    return reply

# --- Chat Input ---
st.subheader("💬 Ask Pathly Anything About Your Career")
user_query = st.chat_input("Type your career question here...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=st.session_state.messages,
        temperature=0.75
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

# --- Initial Trigger ---
if st.sidebar.button("🔄 Generate My Career Plan"):
    if not (name and background and interests):
        st.warning("Please fill in all required fields first.")
    else:
        with st.spinner("🧠 Creating your personalized career roadmap..."):
            generated = generate_initial_plan()
            st.success("Plan generated. Start chatting below or ask for visual charts like a Gantt chart or routine ✨")
