import streamlit as st
from openai import OpenAI
import os
from datetime import datetime
from typing import List

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Pathly - AI Career Coach", layout="wide")
st.title("🤖 Pathly - Your AI Career Coach")
st.markdown("Empowering you with smart, personalized, and visual career advice ✨")

# --- Sidebar: User Input ---
st.sidebar.header("📌 Tell us about yourself")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=12, max_value=70, value=25)
background = st.sidebar.text_area("🧠 Education & Work Background")
interests = st.sidebar.text_area("🎯 Interests & Career Goals")
location = st.sidebar.text_input("🌍 Preferred Work Location (Optional)")
motivation = st.sidebar.slider("🔥 Motivation Level", 0, 10, 7)

# --- Agent Definitions ---
def career_strategist_agent(name, age, background, interests):
    return {
        "role": "user",
        "content": f"""
        You are a senior career strategist. The user named {name}, aged {age}, with the following background:
        {background}, has these goals: {interests}.

        Suggest a detailed career strategy with phases, estimated timelines, skills, and certifications.
        """
    }

def skill_gap_analyzer_agent(background, interests):
    return {
        "role": "user",
        "content": f"""
        You are a skill gap analyst. Given this background:
        {background}, and career interests: {interests}, identify missing skills, tools, or experiences.

        Provide a plan to close the skill gap using online resources or certifications.
        """
    }

def market_analyst_agent(interests, location):
    return {
        "role": "user",
        "content": f"""
        You are a global job market analyst. Given these career interests: {interests}, and location preference: {location},
        identify job roles in demand, companies hiring, and future-proof trends.
        """
    }

def motivational_coach_agent(motivation):
    return {
        "role": "user",
        "content": f"""
        You are a motivational coach. The user's motivation level is {motivation}/10.
        Provide inspiration, habits, mindset tips, and weekly discipline tactics to maintain momentum.
        """
    }

# --- Mermaid Chart Renderer ---
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

# --- Run Multi-Agent System ---
if st.sidebar.button("Generate Personalized Career Advice"):
    if not (name and background and interests):
        st.warning("Please fill in all the required fields.")
    else:
        with st.spinner("🧠 Agents collaborating to build your path..."):
            messages: List[dict] = []

            agents = [
                career_strategist_agent(name, age, background, interests),
                skill_gap_analyzer_agent(background, interests),
                market_analyst_agent(interests, location),
                motivational_coach_agent(motivation),
            ]

            for agent_prompt in agents:
                messages.append(agent_prompt)

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.75
            )

            final_output = response.choices[0].message.content

            if "```mermaid" in final_output:
                roadmap, mermaid_code = final_output.split("```mermaid")
                mermaid_code = mermaid_code.strip().strip("`")
            else:
                roadmap = final_output
                mermaid_code = None

            st.subheader("📋 Complete Career Guidance")
            st.markdown(roadmap)

            if mermaid_code:
                st.subheader("🗺️ Visual Roadmap")
                render_mermaid_chart(mermaid_code)

        st.success("Done! Your custom path is ready 🚀")
