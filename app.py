import os
import json
import datetime
import uuid
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import pytz
import pandas as pd
# Matplotlib and Seaborn are imported but not used in the final code,
# If plots were intended beyond Plotly, ensure they are used or remove imports.
# import matplotlib.pyplot as plt
# import seaborn as sns
import sqlite3
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import requests
import base64 # Imported but not used, remove if not needed.

# --- CONSTANTS ---
APP_NAME = "Aishura"
APP_TAGLINE = "Your AI Career Coach & Growth Partner"
APP_VERSION = "1.0.0"
APP_COLOR_PRIMARY = "#6C63FF"  # Primary brand color (purple)
APP_COLOR_SECONDARY = "#FF6584"  # Secondary brand color (pink-red)
# Consider hosting the logo image or using a more permanent placeholder
APP_LOGO_URL = "https://via.placeholder.com/150x150.png?text=Aishura"

# --- PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title=f"{APP_NAME} - {APP_TAGLINE}",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'mailto:support@aishura.ai',
        'Report a bug': 'mailto:bugs@aishura.ai',
        'About': f"{APP_NAME} {APP_VERSION} - {APP_TAGLINE}"
    }
)

# --- CUSTOM CSS ---
# (CSS remains the same as provided in the file)
st.markdown("""
<style>
    /* Brand colors */
    :root {
        --primary: #6C63FF;
        --secondary: #FF6584;
        --dark: #2E2E48;
        --light: #F8F9FA;
        --success: #28a745;
        --info: #17a2b8;
        --warning: #ffc107;
        --danger: #dc3545;
    }

    /* Header styling */
    .main-header {
        color: var(--primary);
        font-weight: 600;
    }

    /* Improve form elements */
    div[data-testid="stForm"] {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Buttons styling */
    .stButton>button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    /* Primary button */
    .stButton>button[data-baseweb="button"] {
        background-color: var(--primary);
        color: white; /* Ensure text is visible */
    }
    .stButton>button[data-baseweb="button"]:hover {
         background-color: #5850d8; /* Darker primary on hover */
         color: white;
    }

     /* Secondary button */
    .stButton>button:not([data-baseweb="button"]) {
        background-color: var(--light);
        color: var(--primary);
        border: 1px solid var(--primary);
    }
     .stButton>button:not([data-baseweb="button"]):hover {
        background-color: #e2e6ea; /* Light gray on hover */
        color: var(--primary);
        border: 1px solid var(--primary);
    }


    /* Metrics styling */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 15px; /* Increased padding */
        border-radius: 8px; /* Slightly larger radius */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Softer shadow */
        border-left: 4px solid var(--primary); /* Accent border */
        margin-bottom: 1rem; /* Add space below metric */
    }

    /* Card-like containers */
    .card-container {
        background-color: white;
        padding: 25px; /* Increased padding */
        border-radius: 10px;
        box-shadow: 0 6px 10px rgba(0,0,0,0.08); /* Slightly stronger shadow */
        margin-bottom: 25px; /* Increased margin */
        border: 1px solid #eee; /* Subtle border */
    }

    /* Timeline styling for journal entries */
    .timeline-item {
        padding-left: 25px; /* Increased padding */
        border-left: 3px solid var(--primary); /* Thicker border */
        margin-bottom: 20px; /* Increased margin */
        padding-bottom: 20px; /* Increased padding */
        position: relative; /* Needed for pseudo-element */
    }
     /* Add a circle marker to the timeline */
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -9px; /* Adjust based on border-left thickness */
        top: 0;
        width: 15px;
        height: 15px;
        background-color: var(--primary);
        border-radius: 50%;
        border: 2px solid white;
    }


    /* Emotion tags styling */
    .emotion-tag {
        background-color: var(--secondary); /* Changed to secondary color */
        color: white;
        padding: 4px 12px; /* Adjusted padding */
        border-radius: 15px;
        font-size: 0.85rem; /* Slightly larger */
        margin-right: 8px; /* Increased margin */
        display: inline-block; /* Ensure proper spacing */
        margin-bottom: 5px; /* Add margin below tag */
    }

    /* For the dashboard stats */
    .stat-card {
        padding: 20px; /* Increased padding */
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Enhanced shadow */
        text-align: center;
        background-color: white;
        border-top: 4px solid var(--primary); /* Accent top border */
        margin-bottom: 1rem; /* Add space */
    }
     .stat-card h2 {
        margin-bottom: 0.2em !important; /* Adjust spacing */
     }
      .stat-card p {
        color: #555; /* Darker text for readability */
        font-size: 0.9rem;
     }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--dark);
        padding: 15px;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: none;
        margin-bottom: 5px;
    }
     [data-testid="stSidebar"] .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
     [data-testid="stSidebar"] .stButton>button[data-baseweb="button"] { /* Active page button */
        background-color: var(--primary);
        color: white;
     }
     [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white;
     }
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li {
         color: #eee; /* Lighter text for sidebar readability */
      }


    /* Remove padding from containers */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem; /* Add padding for wider layout */
        padding-right: 2rem;
    }

    /* Enhance header margins */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.8em !important; /* Adjust spacing */
        margin-bottom: 0.5em !important;
        color: var(--dark); /* Consistent header color */
    }

    /* Blockquote styling */
     blockquote {
        border-left: 4px solid var(--primary);
        padding-left: 15px;
        margin-left: 0;
        font-style: italic;
        color: #555;
        background-color: #f8f9fa;
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 5px;
     }

     /* Improve list spacing */
      ul, ol {
        padding-left: 25px;
        margin-bottom: 1rem;
      }
      li {
        margin-bottom: 0.5rem;
      }

      /* Progress bar styling */
      .stProgress > div > div > div > div {
            background-color: var(--secondary); /* Use secondary color for progress */
      }
</style>
""", unsafe_allow_html=True)


# --- DATABASE SETUP ---
# (Database setup remains the same as provided in the file)
def init_db():
    """Initialize SQLite database for persistent storage."""
    conn = sqlite3.connect('aishura.db', check_same_thread=False)
    c = conn.cursor()

    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')

    # Create journal entries table
    c.execute('''
    CREATE TABLE IF NOT EXISTS journal_entries (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        entry_text TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Create analysis table
    c.execute('''
    CREATE TABLE IF NOT EXISTS analysis (
        id TEXT PRIMARY KEY,
        entry_id TEXT NOT NULL UNIQUE, -- Ensure only one analysis per entry
        primary_emotion TEXT,
        intensity INTEGER,
        triggers TEXT,
        growth_opportunities TEXT,
        career_stage TEXT, -- Added career_stage
        implied_goals TEXT, -- Added implied_goals
        raw_data TEXT,
        FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
    )
    ''')

    # Create growth plans table
    c.execute('''
    CREATE TABLE IF NOT EXISTS growth_plans (
        id TEXT PRIMARY KEY,
        entry_id TEXT NOT NULL UNIQUE, -- Ensure only one plan per entry
        title TEXT,
        vision_statement TEXT, -- Added vision_statement
        steps TEXT, -- JSON string
        expected_outcomes TEXT, -- JSON string
        potential_obstacles TEXT, -- JSON string
        accountability_tip TEXT, -- Added accountability_tip
        raw_data TEXT,
        FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
    )
    ''')

    # Create action suggestions table
    c.execute('''
    CREATE TABLE IF NOT EXISTS action_suggestions (
        id TEXT PRIMARY KEY,
        entry_id TEXT NOT NULL UNIQUE, -- Ensure only one set of suggestions per entry
        immediate_actions TEXT, -- JSON string
        preparation_guidance TEXT, -- JSON string
        skill_development TEXT, -- JSON string Added skill_development
        key_insight TEXT,
        motivation_quote TEXT, -- Added motivation_quote
        raw_data TEXT,
        FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
    )
    ''')

    # Create industry insights table (Optional, generated on demand in this version)
    # If persistence is needed, uncomment and adapt:
    # c.execute('''
    # CREATE TABLE IF NOT EXISTS industry_insights (
    #     id TEXT PRIMARY KEY,
    #     entry_id TEXT NOT NULL UNIQUE,
    #     industry_identified TEXT,
    #     key_trends TEXT, -- JSON string
    #     in_demand_skills TEXT, -- JSON string
    #     potential_challenges TEXT, -- JSON string
    #     growth_areas TEXT, -- JSON string
    #     career_path_insights TEXT,
    #     raw_data TEXT,
    #     FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
    # )
    # ''')

    conn.commit()
    return conn


# --- CONFIGURATION & API KEY CHECK ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Perform checks AFTER page config
if not GEMINI_API_KEY:
    st.error("🔴 Error: GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Consider making the model configurable or using a standard one like 'gemini-1.5-flash'
    MODEL_NAME = "gemini-1.5-flash" # Use a generally available model
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    # Generation Config (Optional but recommended)
    generation_config = genai.GenerationConfig(
        temperature=0.7, # Adjust creativity/factuality
        top_p=0.9,
        top_k=40,
        # response_mime_type="application/json" # Enforce JSON output if model supports it
    )
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config=generation_config,
        safety_settings=SAFETY_SETTINGS
        )
except Exception as e:
    st.error(f"🔴 Error configuring Gemini API: {e}")
    st.stop()

# --- LLM PROMPTS ---
# (Prompts remain the same as provided in the file)
CAREER_ANALYSIS_PROMPT = """SYSTEM: You are Aishura, an empathetic AI career assistant with expertise in career development, job search strategies, and professional growth. Analyze the user's entry for their primary career-related emotion, intensity (1-10), potential triggers, growth opportunities, career stage, and implied goals. Respond ONLY with a valid JSON object. No explanations or surrounding text.

USER: Analyze this career-related journal entry: {journal_entry_text}

JSON Structure: {
  "primary_emotion": "string - the main emotion they're experiencing about their career (e.g., anxiety, excitement, frustration, confidence)",
  "intensity": integer - value between 1-10 representing intensity of emotion,
  "triggers": ["string - list of 2-3 potential career-related triggers for this emotion"],
  "growth_opportunities": ["string - list of 2-3 growth areas or improvements based on their entry"],
  "career_stage": "string - assessment of where they are in their career (e.g., early-career, mid-career, leadership, transition)",
  "implied_goals": ["string - list of 1-2 implied career goals extracted from their entry"]
}
"""

GROWTH_PLAN_PROMPT = """SYSTEM: You are Aishura, an AI career coach with expertise in professional development. Create a concise, actionable growth plan based on the user's career concern analysis. The plan should be optimistic yet realistic, focused on tangible steps, and incorporate both short-term actions and long-term development. Respond ONLY with a valid JSON object. No explanations or surrounding text.

USER: Create a plan based on this analysis: {analysis_and_goals_json}

JSON Structure: {
  "title": "string - compelling title for the growth plan focused on positive outcomes",
  "vision_statement": "string - aspirational yet achievable vision statement for their career development",
  "steps": [
    {
      "title": "string - clear, action-oriented step title",
      "description": "string - actionable guidance with specifics",
      "timeframe": "string - suggested timeframe (e.g., 'This week', '1-2 months')",
      "success_indicators": ["string - how they'll know they're making progress"]
    }
  ],
  "expected_outcomes": ["string - list of 3-4 positive outcomes if plan is followed"],
  "potential_obstacles": ["string - list of 1-2 potential challenges they might face"],
  "accountability_tip": "string - suggestion for how to stay accountable to this plan"
}
"""

ACTION_SUGGESTION_PROMPT = """SYSTEM: You are Aishura, an AI career assistant providing proactive, actionable career advice. Your guidance should be specific, actionable, and personalized based on the user's current situation (analysis and optional growth plan). Respond ONLY with a valid JSON object. No explanations or surrounding text.

USER: Based on the user's career concern analysis and optionally their growth plan, suggest concrete next steps. Provide practical, immediate actions they can take today.

User Context:
{profile_data_json}

JSON Structure: {
  "immediate_actions": [
    "string - 3-4 specific, small actions the user can take NOW (e.g., 'Update your LinkedIn headline to spotlight your UX design skills', 'Block out 30 minutes to outline talking points for your upcoming performance review')"
  ],
  "preparation_guidance": [
    {
      "item": "string - aspect of their career to prepare (e.g., 'Interview Readiness', 'Resume Enhancement')",
      "guidance": "string - specific, actionable advice for this item"
    }
  ],
  "skill_development": [
    {
      "skill": "string - relevant skill to develop",
      "resource": "string - specific resource type or approach for development (e.g., 'Online course on Coursera', 'Find a mentor', 'Practice project')",
      "why_important": "string - brief explanation of importance to their situation"
    }
  ],
  "key_insight": "string - single, concise, encouraging insight related to their situation",
  "motivation_quote": "string - brief motivational quote relevant to their career situation"
}
"""

INDUSTRY_INSIGHTS_PROMPT = """SYSTEM: You are Aishura, an AI career assistant with expertise across various industries. Provide focused insights about the user's industry or target industry based on their career situation (analysis and optional growth plan). Respond ONLY with a valid JSON object. No explanations or surrounding text.

USER: Based on the user's career information, provide industry-specific insights relevant to their situation:

User Context: {profile_data_json}

JSON Structure: {
  "industry_identified": "string - the industry you've identified from the user's information (or 'General Career Advice' if unclear)",
  "key_trends": ["string - 2-3 current trends in this industry relevant to the user's situation"],
  "in_demand_skills": ["string - 3-4 skills currently valued in this industry that relate to the user's goals"],
  "potential_challenges": ["string - 2 potential industry-specific challenges relevant to the user's situation"],
  "growth_areas": ["string - 2-3 areas within this industry that show promise for growth"],
  "career_path_insights": "string - brief insight about typical or emerging career paths in this industry relevant to the user"
}
"""

# --- SESSION STATE MANAGEMENT ---
def init_state():
    """Initialize session state variables."""
    # Authentication states
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "login" if not st.session_state.authenticated else "main"
    if 'current_entry_id' not in st.session_state:
        st.session_state.current_entry_id = None

    # Feature flags for A/B testing (demo for investors)
    if 'show_industry_insights' not in st.session_state:
        st.session_state.show_industry_insights = True # Default to showing insights
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False

    # Database connection
    if 'db_conn' not in st.session_state:
        st.session_state.db_conn = init_db()

    # Track metrics for demo (Initialize with DB counts)
    if 'total_entries' not in st.session_state:
        st.session_state.total_entries = get_total_entries_count()
    if 'insights_generated' not in st.session_state:
        st.session_state.insights_generated = get_total_insights_count()

# --- DATA PERSISTENCE FUNCTIONS ---
# (Functions remain mostly the same, slight improvements in error handling/saving)
def hash_password(password):
    """Create a secure hash of a password."""
    return hashlib.sha256(password.encode()).hexdigest()

def user_exists(username):
    """Check if a username exists in the database."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    try:
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        return c.fetchone() is not None
    except sqlite3.Error as e:
        st.error(f"Database error checking user: {e}")
        return False # Assume not exists on error

def verify_user(username, password):
    """Verify user credentials."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hashed_password))
        user = c.fetchone()
        if user:
            # Update last login time
            c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user[0],))
            conn.commit()
            return user[0]  # Return user_id
        return None
    except sqlite3.Error as e:
        st.error(f"Database error verifying user: {e}")
        return None

def create_user(username, password, email=None):
    """Create a new user in the database."""
    if user_exists(username):
        return False, "Username already taken"

    conn = st.session_state.db_conn
    c = conn.cursor()
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(password)
    try:
        c.execute(
            "INSERT INTO users (id, username, password, email) VALUES (?, ?, ?, ?)",
            (user_id, username, hashed_password, email)
        )
        conn.commit()
        return True, user_id
    except sqlite3.IntegrityError:
         return False, "Username already exists (concurrent signup attempt?)"
    except sqlite3.Error as e:
        st.error(f"Database error creating user: {e}")
        return False, str(e)

def save_journal_entry(user_id, journal_text):
    """Save a journal entry to the database."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    entry_id = str(uuid.uuid4())

    try:
        c.execute(
            "INSERT INTO journal_entries (id, user_id, entry_text) VALUES (?, ?, ?)",
            (entry_id, user_id, journal_text)
        )
        conn.commit()
        # Update metrics
        st.session_state.total_entries = get_total_entries_count()
        return entry_id
    except sqlite3.Error as e:
        st.error(f"Error saving journal entry: {e}")
        return None

def save_analysis(entry_id, analysis_data):
    """Save analysis results to the database."""
    if not isinstance(analysis_data, dict):
        st.error("Invalid analysis data format (not a dictionary).")
        return False

    conn = st.session_state.db_conn
    c = conn.cursor()
    analysis_id = str(uuid.uuid4())

    try:
        # Convert lists to JSON strings for storage for consistency
        triggers_str = json.dumps(analysis_data.get('triggers', []))
        growth_opps_str = json.dumps(analysis_data.get('growth_opportunities', []))
        implied_goals_str = json.dumps(analysis_data.get('implied_goals', []))

        c.execute(
            """INSERT OR REPLACE INTO analysis -- Use REPLACE to update if entry_id exists
            (id, entry_id, primary_emotion, intensity, triggers, growth_opportunities, career_stage, implied_goals, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id,
                entry_id,
                analysis_data.get('primary_emotion'),
                analysis_data.get('intensity'),
                triggers_str,
                growth_opps_str,
                analysis_data.get('career_stage'),
                implied_goals_str,
                json.dumps(analysis_data) # Store the full original JSON
            )
        )
        conn.commit()
        # Update metrics
        st.session_state.insights_generated = get_total_insights_count()
        return True
    except sqlite3.Error as e:
        st.error(f"Error saving analysis: {e}")
        return False

def save_growth_plan(entry_id, plan_data):
    """Save growth plan to the database."""
    if not isinstance(plan_data, dict):
        st.error("Invalid growth plan data format (not a dictionary).")
        return False

    conn = st.session_state.db_conn
    c = conn.cursor()
    plan_id = str(uuid.uuid4())

    try:
        # Convert complex data to JSON strings for storage
        steps_str = json.dumps(plan_data.get('steps', []))
        outcomes_str = json.dumps(plan_data.get('expected_outcomes', []))
        obstacles_str = json.dumps(plan_data.get('potential_obstacles', []))

        c.execute(
            """INSERT OR REPLACE INTO growth_plans -- Use REPLACE
            (id, entry_id, title, vision_statement, steps, expected_outcomes, potential_obstacles, accountability_tip, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                plan_id,
                entry_id,
                plan_data.get('title'),
                plan_data.get('vision_statement'),
                steps_str,
                outcomes_str,
                obstacles_str,
                plan_data.get('accountability_tip'),
                json.dumps(plan_data)
            )
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Error saving growth plan: {e}")
        return False

def save_action_suggestions(entry_id, action_data):
    """Save action suggestions to the database."""
    if not isinstance(action_data, dict):
        st.error("Invalid action suggestions data format (not a dictionary).")
        return False

    conn = st.session_state.db_conn
    c = conn.cursor()
    action_id = str(uuid.uuid4())

    try:
        # Convert complex data to JSON strings for storage
        immediate_actions_str = json.dumps(action_data.get('immediate_actions', []))
        prep_guidance_str = json.dumps(action_data.get('preparation_guidance', []))
        skill_dev_str = json.dumps(action_data.get('skill_development', []))

        c.execute(
            """INSERT OR REPLACE INTO action_suggestions -- Use REPLACE
            (id, entry_id, immediate_actions, preparation_guidance, skill_development, key_insight, motivation_quote, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                action_id,
                entry_id,
                immediate_actions_str,
                prep_guidance_str,
                skill_dev_str,
                action_data.get('key_insight'),
                action_data.get('motivation_quote'),
                json.dumps(action_data)
            )
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Error saving action suggestions: {e}")
        return False

# Add save_industry_insights if storing them
# def save_industry_insights(entry_id, insights_data): ...

def get_user_entries(user_id, limit=10):
    """Get journal entries for a user with their analysis data (parsed)."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    entries = []
    try:
        c.execute("""
            SELECT j.id, j.entry_text, j.timestamp, a.raw_data
            FROM journal_entries j
            LEFT JOIN analysis a ON j.id = a.entry_id
            WHERE j.user_id = ?
            ORDER BY j.timestamp DESC
            LIMIT ?
        """, (user_id, limit))

        for row in c.fetchall():
            entry_id, entry_text, timestamp, raw_analysis = row
            analysis_dict = None
            if raw_analysis:
                try:
                    analysis_dict = json.loads(raw_analysis)
                except json.JSONDecodeError:
                    # Log error or handle partially corrupted data if needed
                    print(f"Warning: Could not parse analysis JSON for entry {entry_id}")
                    analysis_dict = {"error": "Could not parse analysis data"}

            entries.append({
                "id": entry_id,
                "journal_entry": entry_text,
                "timestamp": timestamp,
                "analysis": analysis_dict # Store the parsed dict or error
            })
    except sqlite3.Error as e:
        st.error(f"Database error fetching user entries: {e}")
    return entries


def get_journal_entry(entry_id):
    """Get a specific journal entry with its parsed analysis."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    entry_data = None
    try:
        c.execute("""
            SELECT j.id, j.user_id, j.entry_text, j.timestamp, a.raw_data
            FROM journal_entries j
            LEFT JOIN analysis a ON j.id = a.entry_id
            WHERE j.id = ?
        """, (entry_id,))

        row = c.fetchone()
        if row:
            entry_id_db, user_id, entry_text, timestamp, raw_analysis = row
            analysis_dict = None
            if raw_analysis:
                try:
                    analysis_dict = json.loads(raw_analysis)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse analysis JSON for entry {entry_id_db}")
                    analysis_dict = {"error": "Could not parse analysis data"}

            entry_data = {
                "id": entry_id_db,
                "user_id": user_id,
                "journal_entry": entry_text,
                "timestamp": timestamp,
                "analysis": analysis_dict
            }
    except sqlite3.Error as e:
        st.error(f"Database error fetching entry {entry_id}: {e}")
    return entry_data


def get_growth_plan(entry_id):
    """Get the parsed growth plan for an entry."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    plan_data = None
    try:
        c.execute("SELECT raw_data FROM growth_plans WHERE entry_id = ?", (entry_id,))
        row = c.fetchone()
        if row and row[0]:
            try:
                plan_data = json.loads(row[0])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse growth plan JSON for entry {entry_id}")
                plan_data = {"error": "Failed to parse growth plan data"}
    except sqlite3.Error as e:
        st.error(f"Database error fetching growth plan for entry {entry_id}: {e}")
        plan_data = {"error": f"DB Error: {e}"} # Return error dict on DB error
    return plan_data # Returns None if not found, dict if found, error dict on failure


def get_action_suggestions(entry_id):
    """Get the parsed action suggestions for an entry."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    action_data = None
    try:
        c.execute("SELECT raw_data FROM action_suggestions WHERE entry_id = ?", (entry_id,))
        row = c.fetchone()
        if row and row[0]:
            try:
                action_data = json.loads(row[0])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse action suggestions JSON for entry {entry_id}")
                action_data = {"error": "Failed to parse action suggestions data"}
    except sqlite3.Error as e:
        st.error(f"Database error fetching action suggestions for entry {entry_id}: {e}")
        action_data = {"error": f"DB Error: {e}"} # Return error dict on DB error
    return action_data # Returns None if not found, dict if found, error dict on failure

# Add get_industry_insights if storing them
# def get_industry_insights(entry_id): ...


def get_total_entries_count(user_id=None):
    """Get total number of entries for analytics (optionally per user)."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    try:
        if user_id:
            c.execute("SELECT COUNT(*) FROM journal_entries WHERE user_id = ?", (user_id,))
        else: # System-wide count
            c.execute("SELECT COUNT(*) FROM journal_entries")
        result = c.fetchone()
        return result[0] if result else 0
    except sqlite3.Error as e:
        st.error(f"Database error counting entries: {e}")
        return 0


def get_total_insights_count(user_id=None):
    """Get total number of insights generated (optionally per user)."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    try:
        if user_id:
            c.execute("""
                SELECT COUNT(a.id) FROM analysis a
                JOIN journal_entries j ON a.entry_id = j.id
                WHERE j.user_id = ?
            """, (user_id,))
        else: # System-wide count
            c.execute("SELECT COUNT(*) FROM analysis")
        result = c.fetchone()
        return result[0] if result else 0
    except sqlite3.Error as e:
        st.error(f"Database error counting insights: {e}")
        return 0


def get_user_emotion_distribution(user_id, limit=20):
    """Get distribution of primary emotions for visualization."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    emotions = []
    counts = []
    try:
        c.execute("""
            SELECT a.primary_emotion, COUNT(*) as count
            FROM analysis a
            JOIN journal_entries j ON a.entry_id = j.id
            WHERE j.user_id = ? AND a.primary_emotion IS NOT NULL AND a.primary_emotion != ''
            GROUP BY a.primary_emotion
            ORDER BY count DESC
            LIMIT ?
        """, (user_id, limit))

        for emotion, count in c.fetchall():
            emotions.append(emotion.capitalize()) # Capitalize for display
            counts.append(count)
    except sqlite3.Error as e:
        st.error(f"Database error fetching emotion distribution: {e}")
    return emotions, counts

def get_user_intensity_trend(user_id, limit=30):
    """Get intensity over time for trend visualization."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    timestamps = []
    intensities = []
    try:
        c.execute("""
            SELECT j.timestamp, a.intensity
            FROM analysis a
            JOIN journal_entries j ON a.entry_id = j.id
            WHERE j.user_id = ? AND a.intensity IS NOT NULL
            ORDER BY j.timestamp ASC -- Chronological order
            LIMIT ?
        """, (user_id, limit))

        for timestamp, intensity in c.fetchall():
            # Convert timestamp string to datetime object for plotting if needed
            try:
                # Attempt parsing with microseconds
                 dt_obj = datetime.datetime.fromisoformat(timestamp.replace(' ', 'T'))
            except ValueError:
                 # Fallback if microseconds are missing
                 dt_obj = datetime.datetime.strptime(timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                 
            timestamps.append(dt_obj)
            intensities.append(intensity)

    except sqlite3.Error as e:
        st.error(f"Database error fetching intensity trend: {e}")
    return timestamps, intensities


# --- LLM INTERACTION LOGIC ---
def call_gemini_llm(prompt: str, expecting_json: bool = True) -> Dict | str | None:
    """Call the Gemini LLM with error handling and optional JSON parsing."""
    try:
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)

        # Check for safety blocks or empty response
        if not response.candidates:
            feedback = getattr(response, 'prompt_feedback', None)
            block_reason = getattr(feedback, 'block_reason', 'Unknown') if feedback else 'Unknown'
            finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown') if hasattr(response, 'candidates') and response.candidates else 'Unknown' # Added safety check
            error_msg = f"Content generation failed. Reason: {block_reason} / Finish Reason: {finish_reason}."
            st.error(f"LLM Error: {error_msg}")
            return {"error": error_msg, "raw_response": None} # Return error dict

        raw_text = response.text.strip()

        if not expecting_json:
             return raw_text # Return raw text if JSON is not expected

        # Attempt to parse JSON
        try:
            # Remove potential markdown fences
            if raw_text.startswith("```json"):
                cleaned_text = raw_text.removeprefix("```json").removesuffix("```").strip()
            elif raw_text.startswith("```"):
                 cleaned_text = raw_text.removeprefix("```").removesuffix("```").strip()
            else:
                 cleaned_text = raw_text

            parsed_json = json.loads(cleaned_text)
            return parsed_json # Return parsed dictionary
        except json.JSONDecodeError as json_err:
            error_msg = f"Failed to parse LLM response as JSON: {json_err}"
            st.warning(f"LLM JSON Parsing Error: {error_msg}. Raw response:\n```\n{raw_text}\n```")
            # Return error dict with raw response for debugging
            return {"error": error_msg, "raw_response": raw_text}
        except Exception as parse_err:
             error_msg = f"Unexpected error processing LLM response: {parse_err}"
             st.error(f"LLM Processing Error: {error_msg}")
             return {"error": error_msg, "raw_response": raw_text}

    except Exception as e:
        # Catch potential API call errors (network, auth, etc.)
        error_msg = f"Gemini API call failed: {e}"
        st.error(error_msg)
        return {"error": error_msg, "raw_response": None} # Return error dict


def analyze_career_entry(journal_entry: str) -> Dict:
    """Analyze a career journal entry using the LLM."""
    prompt = CAREER_ANALYSIS_PROMPT.format(journal_entry_text=journal_entry)
    result = call_gemini_llm(prompt, expecting_json=True)
    # Ensure result is always a dict, even on non-JSON or error returns
    return result if isinstance(result, dict) else {"error": "Invalid response type from LLM", "raw_response": str(result)}


def generate_growth_plan(emotion_analysis: Dict) -> Dict:
    """Generate a personalized growth plan."""
    if not emotion_analysis or not isinstance(emotion_analysis, dict) or 'error' in emotion_analysis:
        st.warning("Cannot generate plan without valid prior analysis.")
        return {"error": "Valid analysis required to generate growth plan."}

    # Remove raw_response if present in the analysis dict before sending
    analysis_for_prompt = {k: v for k, v in emotion_analysis.items() if k != 'raw_response'}
    input_json_str = json.dumps(analysis_for_prompt, indent=2)

    prompt = GROWTH_PLAN_PROMPT.format(analysis_and_goals_json=input_json_str)
    result = call_gemini_llm(prompt, expecting_json=True)
    return result if isinstance(result, dict) else {"error": "Invalid response type from LLM", "raw_response": str(result)}


def generate_action_suggestions(emotion_analysis: Dict, growth_plan: Optional[Dict] = None) -> Dict:
    """Generate actionable career suggestions."""
    if not emotion_analysis or not isinstance(emotion_analysis, dict) or 'error' in emotion_analysis:
        st.warning("Cannot generate actions without valid prior analysis.")
        return {"error": "Valid analysis required to generate actions."}

    # Prepare context, removing potential error keys or raw responses
    analysis_for_prompt = {k: v for k, v in emotion_analysis.items() if k != 'raw_response' and k != 'error'}
    plan_for_prompt = {k: v for k, v in growth_plan.items() if k != 'raw_response' and k != 'error'} if growth_plan and isinstance(growth_plan, dict) else {"message": "No growth plan available or plan contains errors."}

    profile_data = {
        "emotion_analysis": analysis_for_prompt,
        "growth_plan": plan_for_prompt,
    }

    input_json_str = json.dumps(profile_data, indent=2)
    prompt = ACTION_SUGGESTION_PROMPT.format(profile_data_json=input_json_str)
    result = call_gemini_llm(prompt, expecting_json=True)
    return result if isinstance(result, dict) else {"error": "Invalid response type from LLM", "raw_response": str(result)}


def generate_industry_insights(emotion_analysis: Dict, growth_plan: Optional[Dict] = None) -> Dict:
    """Generate industry-specific insights."""
    if not emotion_analysis or not isinstance(emotion_analysis, dict) or 'error' in emotion_analysis:
         st.warning("Cannot generate industry insights without valid prior analysis.")
         return {"error": "Valid analysis required to generate industry insights."}

    # Prepare context, removing potential error keys or raw responses
    analysis_for_prompt = {k: v for k, v in emotion_analysis.items() if k != 'raw_response' and k != 'error'}
    plan_for_prompt = {k: v for k, v in growth_plan.items() if k != 'raw_response' and k != 'error'} if growth_plan and isinstance(growth_plan, dict) else {"message": "No growth plan available or plan contains errors."}

    profile_data = {
        "emotion_analysis": analysis_for_prompt,
        "growth_plan": plan_for_prompt,
    }

    input_json_str = json.dumps(profile_data, indent=2)
    prompt = INDUSTRY_INSIGHTS_PROMPT.format(profile_data_json=input_json_str)
    result = call_gemini_llm(prompt, expecting_json=True)
    return result if isinstance(result, dict) else {"error": "Invalid response type from LLM", "raw_response": str(result)}


# --- UI COMPONENTS ---
def render_brand_header():
    """Render the brand header/logo area."""
    col1, col2 = st.columns([1, 6]) # Adjust ratio if needed
    with col1:
        # Consider adding error handling for image loading
        try:
            # You might want to download the image once or use a local path
            st.image(APP_LOGO_URL, width=80)
        except Exception as img_err:
            st.warning(f"Could not load logo: {img_err}")
            st.write(f"**{APP_NAME[0]}**") # Fallback to initial
    with col2:
        st.markdown(f"<h1 class='main-header' style='margin-bottom: 0.1em;'>{APP_NAME}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin-top: 0; color: #555;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
    st.markdown("---") # Add a separator


def render_loading_animation(text="Processing..."):
    """Render a custom loading animation for AI processing."""
    loader_html = f"""
    <div style="display:flex; flex-direction: column; align-items:center; justify-content:center; margin:2rem 0; text-align:center;">
        <div style="border: 6px solid #f3f3f3; border-top: 6px solid var(--primary); border-radius:50%; width:50px; height:50px; animation:spin 1s linear infinite;"></div>
        <p style="margin-top: 1rem; color: var(--dark);">{text}</p>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    st.markdown(loader_html, unsafe_allow_html=True)


def render_login_page():
    """Render the login/signup page."""
    # Brand header centered for login page
    st.markdown(f"<div style='text-align:center;'><img src='{APP_LOGO_URL}' width='100'></div>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='main-header' style='text-align:center;'>Welcome to {APP_NAME}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size: 1.1rem; color: #444;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) # Add space

    # Feature highlights
    cols = st.columns(3)
    features = [
        ("✨ Smart Career Analysis", "AI-powered insights to understand your current situation and identify growth opportunities."),
        ("🚀 Growth Planning", "Personalized roadmaps to help you achieve your professional goals step-by-step."),
        ("💡 Actionable Guidance", "Practical suggestions and industry insights to accelerate your career progress.")
    ]
    for col, (title, text) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="card-container" style="text-align:center; height: 180px;">
                <h4>{title}</h4>
                <p style="font-size: 0.9rem;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Login/Signup columns
    col1, col2 = st.columns([1, 1]) # Equal width columns

    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("💼 Login")
        with st.form("login_form_aishura"):
            uname = st.text_input("Username", key="login_u_aishura")
            pwd = st.text_input("Password", type="password", key="login_p_aishura")
            login_col1, login_col2 = st.columns([3, 2]) # Adjust button width ratio
            with login_col1:
                login_btn = st.form_submit_button("Login", use_container_width=True, type="primary")
            with login_col2:
                demo_btn = st.form_submit_button("Try Demo", use_container_width=True) # Changed text

            if login_btn:
                if not uname or not pwd:
                    st.error("Please enter both username and password.")
                else:
                    user_id = verify_user(uname, pwd)
                    if user_id:
                        st.session_state.current_user = uname
                        st.session_state.current_user_id = user_id
                        st.session_state.authenticated = True
                        st.session_state.current_view = "main"
                        st.session_state.demo_mode = False # Ensure demo mode is off
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            if demo_btn:
                # Use predefined demo credentials or create a temporary one
                st.session_state.current_user = "Demo User"
                st.session_state.current_user_id = "demo-user-001" # Consistent demo ID
                st.session_state.authenticated = True
                st.session_state.current_view = "main"
                st.session_state.demo_mode = True
                 # Optional: Pre-populate demo data if needed here
                st.success("Entering Demo Mode...")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("✨ Sign Up")
        with st.form("signup_form_aishura"):
            s_uname = st.text_input("Choose Username", key="signup_u_aishura")
            s_email = st.text_input("Email (optional)", key="signup_e_aishura")
            s_pwd = st.text_input("Choose Password (min 6 chars)", type="password", key="signup_p_aishura")
            s_pwd_confirm = st.text_input("Confirm Password", type="password", key="signup_pc_aishura")
            signup_btn = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if signup_btn:
                if not s_uname or not s_pwd:
                    st.warning("Username and password are required.")
                elif s_pwd != s_pwd_confirm:
                    st.error("Passwords do not match.")
                elif len(s_pwd) < 6:
                    st.error("Password must be at least 6 characters long.")
                else:
                    # Basic email format check (optional but good)
                    if s_email and "@" not in s_email:
                         st.error("Please enter a valid email address or leave it blank.")
                    else:
                         success, result = create_user(s_uname, s_pwd, s_email if s_email else None)
                         if success:
                             st.success("Account created successfully! Please log in.")
                             # Optionally clear the form fields here
                         else:
                             st.error(f"Failed to create account: {result}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center; margin-top:40px; font-size:0.9rem; color: #777;">
        <p>© {current_year} {app_name} - {tagline}</p>
    </div>
    """.format(current_year=datetime.datetime.now().year, app_name=APP_NAME, tagline=APP_TAGLINE), unsafe_allow_html=True)


def render_sidebar():
    """Render the navigation sidebar."""
    if not st.session_state.get('authenticated'):
        return # Don't render sidebar if not logged in

    with st.sidebar:
        # Simplified header in sidebar
        st.image(APP_LOGO_URL, width=60)
        st.title(f"{APP_NAME}")
        st.markdown("---")

        user_name = st.session_state.get('current_user', 'User')
        demo_badge = " (Demo Mode)" if st.session_state.get('demo_mode', False) else ""
        st.markdown(f"#### Welcome, {user_name}{demo_badge}")

        st.markdown("---")

        # Navigation menu using radio buttons for better active state handling
        st.subheader("Navigation")
        pages = {
            "🏠 Dashboard": "main",
            "📝 Career Journal": "journal",
            "📊 Progress & Insights": "progress"
        }
        # Get index of current view for radio default
        view_ids = list(pages.values())
        current_view_id = st.session_state.get('current_view', 'main')
        try:
            current_index = view_ids.index(current_view_id) if current_view_id in view_ids else 0
        except ValueError:
             current_index = 0 # Default to dashboard if view is invalid (e.g., analysis)

        selected_page_label = st.radio(
             "Go to",
             options=list(pages.keys()),
             index=current_index,
             key="sidebar_nav_radio",
             label_visibility="collapsed" # Hide the 'Go to' label
        )

        # Check if selection changed and update view
        new_view_id = pages[selected_page_label]
        if new_view_id != current_view_id:
             st.session_state.current_view = new_view_id
             # Reset entry ID when navigating away from analysis view
             if 'current_entry_id' in st.session_state and current_view_id == 'analysis':
                 del st.session_state['current_entry_id']
             st.rerun()


        st.markdown("---")

        # Account/Settings section
        st.subheader("Account") # Changed header level
        if st.button("⚙️ Settings", key="settings_btn", use_container_width=True):
            st.session_state.current_view = "settings"
            # Reset entry ID if coming from analysis
            if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
            st.rerun()

        if st.button("🔒 Logout", key="logout_aishura", use_container_width=True):
            # Clear sensitive session state keys upon logout
            keys_to_clear = ['authenticated', 'current_user', 'current_user_id', 'current_view', 'current_entry_id', 'demo_mode']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Set view to login explicitly AFTER clearing state
            st.session_state.current_view = "login"
            st.success("Logged out successfully.")
            st.rerun()

        # Demo info
        if st.session_state.get('demo_mode', False):
            st.markdown("---")
            st.info("This is a demonstration version. Data is not persistently saved across sessions in demo mode.")

        # Version info
        st.markdown("---")
        st.caption(f"Version {APP_VERSION}")


def render_main_dashboard():
    """Render the main dashboard view."""
    render_brand_header() # Use the standard header
    st.title("✨ Your Career Dashboard")

    user_id = st.session_state.get('current_user_id')
    if not user_id:
        st.error("User ID not found. Please log in again.")
        st.session_state.current_view = "login"
        st.rerun()
        return

    # Get data needed for dashboard
    entry_history = get_user_entries(user_id, limit=20) # Get more for better stats/viz
    total_user_entries = get_total_entries_count(user_id) # Get specific count for user

    # Welcome message and quick action
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### How can I help you today, {st.session_state.get('current_user', 'User')}?")
    with col2:
        if st.button("➕ New Reflection", type="primary", use_container_width=True, key="new_entry_dash_aishura"):
            st.session_state.current_view = "journal"
            if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Your Progress Overview")

    # Calculate stats
    top_emotion = "N/A"
    valid_analysis_count = 0
    intensities = []
    plans_count = 0

    if entry_history:
        emotions_counter = {}
        for entry in entry_history:
            analysis = entry.get('analysis')
            if analysis and isinstance(analysis, dict) and 'error' not in analysis:
                valid_analysis_count += 1
                if emotion := analysis.get('primary_emotion'):
                     emotions_counter[emotion] = emotions_counter.get(emotion, 0) + 1
                if intensity := analysis.get('intensity'):
                     intensities.append(intensity)

                 # Check for growth plan associated with this entry
                plan = get_growth_plan(entry['id']) # Check DB directly
                if plan and 'error' not in plan:
                    plans_count += 1

        if emotions_counter:
            top_emotion = max(emotions_counter, key=emotions_counter.get).capitalize()

    avg_intensity = f"{sum(intensities) / len(intensities):.1f}/10" if intensities else "N/A"

    # Display Stats in cards
    stat1, stat2, stat3, stat4 = st.columns(4)
    stats_data = [
        (stat1, total_user_entries, "Reflections Made"),
        (stat2, top_emotion, "Common Theme"),
        (stat3, avg_intensity, "Avg. Intensity"),
        (stat4, plans_count, "Growth Plans")
    ]

    colors = [APP_COLOR_PRIMARY, APP_COLOR_SECONDARY, APP_COLOR_PRIMARY, APP_COLOR_SECONDARY] # Alternate colors

    for i, (col, value, label) in enumerate(stats_data):
         with col:
            st.markdown(f"""
            <div class="stat-card" style="border-top-color: {colors[i]};">
                <h2 style="color:{colors[i]}; margin:0;">{value}</h2>
                <p style="margin:0;">{label}</p>
            </div>
            """, unsafe_allow_html=True)


    # Recent entries section
    st.markdown("---")
    st.subheader("⏳ Recent Career Insights")

    if not entry_history:
        st.info("📝 Click 'New Reflection' above to share how you're feeling about your career or what challenges you're facing. Aishura will provide personalized insights.")
    else:
        st.markdown("Click 'View Full Analysis' to see detailed insights, growth plans, and action suggestions for each entry.")
        # Display the most recent 3 entries with more details
        for i, entry in enumerate(entry_history[:3]): # Show top 3
            entry_id = entry.get('id')
            ts_str = entry.get('timestamp', '')
            try:
                # Try parsing with microseconds, fallback if needed
                 dt_obj = datetime.datetime.fromisoformat(ts_str.replace(' ', 'T')) if ts_str else None
                 display_time = dt_obj.strftime('%Y-%m-%d %H:%M') if dt_obj else 'Unknown time'
            except ValueError:
                 display_time = ts_str[:16] # Fallback display

            analysis = entry.get('analysis', {})
            is_analysis_ok = isinstance(analysis, dict) and 'error' not in analysis
            p_emotion = analysis.get('primary_emotion', 'N/A') if is_analysis_ok else 'N/A'
            intensity = f"({analysis.get('intensity', '?')}/10)" if is_analysis_ok and 'intensity' in analysis else ""
            career_stage = analysis.get('career_stage', 'Unknown Stage') if is_analysis_ok else 'Unknown Stage'

            with st.container(): # Use container for better separation
                 st.markdown(f'<div class="card-container">', unsafe_allow_html=True) # Wrap in card

                 col1, col2 = st.columns([4, 1])
                 with col1:
                    st.markdown(f"""
                        <span class="emotion-tag">{p_emotion.capitalize()} {intensity}</span>
                        <span style="color:#555; font-size:0.9rem;"> | Stage: {career_stage.capitalize()}</span>
                        <br>
                        <span style="color:#666; font-size:0.8rem;">{display_time}</span>
                    """, unsafe_allow_html=True)
                 with col2:
                    if st.button("View Full Analysis", key=f"view_dash_aishura_{entry_id}", use_container_width=True):
                        st.session_state.current_entry_id = entry_id
                        st.session_state.current_view = "analysis"
                        st.rerun()


                 st.markdown(f"**Your Reflection:**")
                 st.markdown(f"> {entry.get('journal_entry', 'N/A')[:200] + ('...' if len(entry.get('journal_entry', 'N/A')) > 200 else '')}") # Show more text

                 # Show brief analysis summary if available
                 if is_analysis_ok:
                     triggers = analysis.get('triggers', [])
                     opps = analysis.get('growth_opportunities', [])
                     if triggers or opps:
                          st.markdown("**Key Points:**")
                          summary_points = []
                          if triggers: summary_points.append(f"Triggers: {', '.join(triggers[:2])}{'...' if len(triggers) > 2 else ''}")
                          if opps: summary_points.append(f"Opportunities: {', '.join(opps[:2])}{'...' if len(opps) > 2 else ''}")
                          st.markdown(f"<ul style='margin-bottom: 0;'>{''.join(f'<li>{p}</li>' for p in summary_points)}</ul>", unsafe_allow_html=True)


                 st.markdown("</div>", unsafe_allow_html=True) # Close card


    # Visualization section (if user has enough data)
    if valid_analysis_count >= 3: # Require at least 3 entries with valid analysis
        st.markdown("---")
        st.subheader("📈 Your Career Emotion Trends")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
             st.markdown("**Emotion Frequency**")
             emotions, counts = get_user_emotion_distribution(user_id)
             if emotions and counts:
                # Create a Plotly pie chart for emotion distribution
                fig_pie = px.pie(
                    names=emotions,
                    values=counts,
                    title="Distribution of Primary Emotions",
                    hole=0.3, # Donut chart style
                    color_discrete_sequence=px.colors.sequential.Purples_r # Use purple scale
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    showlegend=False,
                    margin=dict(t=50, b=0, l=0, r=0), # Adjust margins
                    title_x=0.5, # Center title
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#333")
                )
                st.plotly_chart(fig_pie, use_container_width=True)
             else:
                 st.info("Not enough emotion data yet to show distribution.")

        with viz_col2:
             st.markdown("**Intensity Over Time**")
             timestamps, intensities_trend = get_user_intensity_trend(user_id, limit=30) # Get last 30
             if timestamps and intensities_trend and len(timestamps) > 1:
                 # Create a Plotly line chart for intensity trend
                 df_trend = pd.DataFrame({'Timestamp': timestamps, 'Intensity': intensities_trend})
                 fig_line = px.line(
                    df_trend,
                    x='Timestamp',
                    y='Intensity',
                    title="Intensity of Reflections Over Time",
                    markers=True,
                    line_shape='spline' # Smoothed line
                 )
                 fig_line.update_layout(
                    yaxis_range=[0, 10.5], # Set Y-axis from 0 to 10
                    xaxis_title="",
                    yaxis_title="Intensity (1-10)",
                    margin=dict(t=50, b=20, l=40, r=20),
                    title_x=0.5,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#f8f9fa', # Light background for plot area
                    font=dict(color="#333")
                 )
                 fig_line.update_traces(line=dict(color=APP_COLOR_SECONDARY)) # Use secondary color
                 st.plotly_chart(fig_line, use_container_width=True)
             else:
                 st.info("Not enough intensity data yet to show trend (need at least 2 entries).")

    else:
        st.info("💡 Keep reflecting on your career! Once you have more entries with analysis, you'll see trend visualizations here.")


def render_journal_page():
    """Render the career journal entry page."""
    render_brand_header()
    st.title("📝 New Career Reflection")
    st.subheader("Share your thoughts, feelings, or challenges about your career")

    st.markdown("""
    <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px; border-left: 5px solid var(--primary);">
        <p><strong>How to use this space:</strong></p>
        <ul style="margin-bottom: 0;">
            <li>Reflect on your current job satisfaction or dissatisfaction.</li>
            <li>Describe challenges in your job search or application process.</li>
            <li>Outline your thoughts on a potential career change or transition.</li>
            <li>Detail skills you want to develop or career goals you're pursuing.</li>
            <li>Share feelings about upcoming interviews, reviews, or projects.</li>
        </ul>
        <p style="margin-top: 10px;"><em>The more detail you provide, the better Aishura can understand and assist you.</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Journal entry form
    with st.form("journal_form_aishura"):
        journal_entry = st.text_area(
            "Your career thoughts:", # Simpler label
            height=200, # Slightly taller
            key="journal_input_aishura",
            placeholder="What's on your mind regarding your career? (e.g., 'Feeling stuck in my current role...', 'Excited about a new project leadership opportunity...', 'Nervous about negotiating salary for a job offer...')"
        )

        # Demo examples (only show if demo mode is active)
        if st.session_state.get('demo_mode', False):
            st.caption("Demo examples (click to fill):")
            examples = {
                "Job Search Frustration": "I've been applying to data scientist positions for three months now with no luck. I've had a few interviews but nothing has converted to an offer. I'm starting to doubt my skills and wonder if I need to learn more tools or languages. My resume seems strong but maybe it's not ATS-friendly?",
                "Leadership Aspiration": "I'm a senior software engineer and really want to move into a team lead or management position within the next year. My technical skills are strong, but I'm not sure how to best demonstrate leadership potential or what specific skills I should focus on developing. Should I talk to my manager?",
                "Career Change Uncertainty": "After 8 years in marketing, I'm seriously considering switching to UX/UI design. I'm passionate about the creative aspects and user focus, but nervous about starting over and taking a potential pay cut. I've done some online courses, but how do I build a portfolio and make the leap realistically?"
            }
            example_cols = st.columns(len(examples))
            for i, (label, text) in enumerate(examples.items()):
                with example_cols[i]:
                    if st.button(label, key=f"example_{i+1}", use_container_width=True):
                        st.session_state.journal_input_aishura = text
                        st.rerun() # Rerun to update the text area

        # Form submission buttons
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col1:
            submit_btn = st.form_submit_button("🚀 Analyze & Get Insights", type="primary", use_container_width=True)
        with submit_col2:
            clear_btn = st.form_submit_button("Clear", use_container_width=True)

        if clear_btn:
            st.session_state.journal_input_aishura = ""
            st.rerun()

        if submit_btn:
            entry_text = journal_entry.strip()
            if not entry_text or len(entry_text) < 20: # Increased minimum length
                st.warning("Please provide a bit more detail about your career situation (at least 20 characters) for a meaningful analysis.")
            else:
                # Show loading animation while processing
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                     render_loading_animation("Analyzing your reflection...")

                # Call LLM for analysis
                analysis_result = analyze_career_entry(entry_text)

                loading_placeholder.empty() # Clear loading animation

                # Process result
                if isinstance(analysis_result, dict) and 'error' not in analysis_result:
                    # Save journal entry first
                    entry_id = save_journal_entry(st.session_state.current_user_id, entry_text)
                    if entry_id:
                        # Save the successful analysis linked to the entry
                        if save_analysis(entry_id, analysis_result):
                            st.session_state.current_entry_id = entry_id
                            st.session_state.current_view = "analysis"
                            st.success("Analysis complete! Redirecting...")
                            st.rerun()
                        else:
                            # Analysis saving failed (DB error?), inform user
                            # Entry is saved, but analysis isn't. Might need manual retry option later.
                            st.error("Failed to save the analysis results. Your entry was saved, but insights might be missing. Please try analyzing again later or contact support.")
                    else:
                        # Entry saving failed (DB error?), inform user
                        st.error("Failed to save your journal entry. Please try again.")
                else:
                    # Analysis failed (LLM error or JSON parse error)
                    error_msg = analysis_result.get('error', 'Unknown analysis error') if isinstance(analysis_result, dict) else "Unknown analysis error"
                    st.error(f"Could not analyze your entry: {error_msg}. Please try rephrasing or try again later.")
                    # Optionally show raw response if available and useful for debugging
                    if isinstance(analysis_result, dict) and 'raw_response' in analysis_result and analysis_result['raw_response']:
                        st.expander("Show Raw Response (for debugging)").code(analysis_result['raw_response'])

    # Past entries section (reuse logic from dashboard but potentially show more)
    st.markdown("---")
    st.subheader("🕰️ Your Reflection History")

    user_id = st.session_state.current_user_id
    history = get_user_entries(user_id, limit=10) # Show more history on this page

    if not history:
        st.info("You haven't created any journal entries yet. Use the form above to start!")
    else:
        st.markdown("Review your past reflections and click 'View' to see the detailed analysis.")
        for i, entry in enumerate(history):
             entry_id = entry.get('id')
             ts_str = entry.get('timestamp', '')
             try:
                 dt_obj = datetime.datetime.fromisoformat(ts_str.replace(' ', 'T')) if ts_str else None
                 display_time = dt_obj.strftime('%Y-%m-%d %H:%M') if dt_obj else 'Unknown time'
             except ValueError:
                 display_time = ts_str[:16]

             analysis = entry.get('analysis', {})
             is_analysis_ok = isinstance(analysis, dict) and 'error' not in analysis
             p_emotion = analysis.get('primary_emotion', 'N/A').capitalize() if is_analysis_ok else 'N/A'
             intensity = f"({analysis.get('intensity', '?')}/10)" if is_analysis_ok and 'intensity' in analysis else ""

             # Use timeline item styling
             st.markdown(f'<div class="timeline-item">', unsafe_allow_html=True)
             col1, col2 = st.columns([4, 1])
             with col1:
                 st.markdown(f"""
                     <p style="margin:0;">
                         <strong style="color:var(--primary);">{display_time}</strong>
                         <span class="emotion-tag" style="margin-left: 10px;">{p_emotion} {intensity}</span>
                     </p>
                     <p style="margin-top:5px; color: #444;">{entry.get('journal_entry', '')[:120]}...</p>
                 """, unsafe_allow_html=True)
             with col2:
                  # Use a less prominent button style for list view
                  if st.button("View Details", key=f"list_journ_aishura_{entry_id}", use_container_width=True):
                      st.session_state.current_entry_id = entry_id
                      st.session_state.current_view = "analysis"
                      st.rerun()
             st.markdown(f'</div>', unsafe_allow_html=True)
             # Add a subtle separator between entries if desired
             # if i < len(history) - 1: st.markdown("<hr style='border-top: 1px solid #eee;'>", unsafe_allow_html=True)


def render_analysis_page(entry_id):
    """Render the analysis page for a specific journal entry."""
    render_brand_header()
    st.title("💬 Career Insight Analysis")

    user_id = st.session_state.get('current_user_id')
    entry_data = get_journal_entry(entry_id) # Fetches entry + analysis

    if not entry_data:
        st.error(f"Error: Could not load reflection with ID {entry_id}.")
        if st.button("⬅️ Back to Journal", key="back_analysis_err_aishura"):
            st.session_state.current_view = "journal"
            if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
            st.rerun()
        return

    # Security check: Ensure the entry belongs to the logged-in user
    if entry_data.get('user_id') != user_id and not st.session_state.get('demo_mode'): # Allow demo users to see anything for now
         st.error("Unauthorized access to this reflection.")
         st.session_state.current_view = "main" # Redirect to dashboard
         if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
         st.rerun()
         return

    # --- Display Original Entry ---
    st.markdown("### Your Reflection")
    try:
        dt_obj = datetime.datetime.fromisoformat(entry_data.get('timestamp', '').replace(' ', 'T'))
        display_time = dt_obj.strftime('%B %d, %Y at %I:%M %p') # More readable format
    except:
        display_time = entry_data.get('timestamp', 'Unknown time')

    st.markdown(f"""
    <div class="card-container" style="background-color: #f8f9fa;">
        <p style="text-align:right; font-size:0.8rem; color:#666; margin-bottom: 5px;">
            Shared on: {display_time}
        </p>
        <blockquote>
            {entry_data.get('journal_entry', 'N/A')}
        </blockquote>
    </div>
    """, unsafe_allow_html=True)


    # --- Display Analysis ---
    st.markdown("### 💡 Aishura's Analysis")
    analysis = entry_data.get('analysis') # This is already parsed or contains error dict
    analysis_placeholder = st.empty() # Placeholder for analysis content or generation button

    if not analysis:
        # This case shouldn't happen if save_analysis is robust, but handle it.
         with analysis_placeholder.container():
             st.warning("Analysis data is missing for this entry.")
             # Optionally add a button to re-analyze
             if st.button("Re-analyze Entry?", key="reanalyze_missing"):
                  # Add re-analysis logic here...
                  st.info("Re-analysis feature not yet implemented.")

    elif 'error' in analysis:
         with analysis_placeholder.container():
             st.error(f"Analysis Error: {analysis['error']}")
             if raw := analysis.get('raw_response'):
                 st.expander("Show Raw Debug Info").code(raw)
             # Optionally add a button to re-analyze
             if st.button("Try Analyzing Again?", key="reanalyze_error"):
                  # Add re-analysis logic here...
                   st.info("Re-analysis feature not yet implemented.")

    else: # Analysis exists and is valid
         with analysis_placeholder.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)

            # Empathetic opening based on emotion
            primary_emotion = analysis.get('primary_emotion', '').lower()
            career_stage = analysis.get('career_stage', 'your career situation').lower()
            empathetic_responses = { # Expanded responses
                'anxious': f"It sounds like you're feeling quite **anxious** about {career_stage}. That's understandable given the pressures involved. Let's break down these feelings and find actionable steps to build confidence.",
                'frustrated': f"I hear your **frustration** regarding {career_stage}. It's challenging when things don't go as planned. Let's explore the root causes and identify strategies to move past these roadblocks.",
                'excited': f"That's wonderful you're feeling **excited** about {career_stage}! This positive energy is a great asset. Let's leverage this enthusiasm to create a clear plan for your next steps.",
                'confused': f"Feeling **confused** about {career_stage} is common. There are often many paths and possibilities. Let's work together to bring clarity to your options and goals.",
                'overwhelmed': f"It seems you're feeling **overwhelmed** by {career_stage}. This can happen when juggling multiple demands. Let's focus on prioritizing and breaking things down into manageable parts.",
                'confident': f"It's great that you're feeling **confident** about {career_stage}. This self-assurance is valuable. Let's ensure you have the right strategies to maintain this momentum and achieve your goals.",
                'hopeful': f"Detecting a sense of **hope** regarding {career_stage} is encouraging. Hope fuels action. Let's translate this feeling into concrete steps towards the future you envision.",
                'discouraged': f"Feeling **discouraged** with {career_stage} can be tough. Remember that setbacks are part of growth. Let's identify small wins to rebuild momentum and find new perspectives.",
                'uncertain': f"Navigating **uncertainty** in {career_stage} is a frequent challenge. Acknowledging this feeling is the first step. Let's explore ways to gather information and reduce ambiguity.",
                'stuck': f"Feeling **stuck** in {career_stage} can be demotivating. Let's identify what's holding you back and brainstorm potential pathways to get you moving forward again.",
                'motivated': f"It's great you're feeling **motivated** regarding {career_stage}! Let's harness this drive with a clear plan to ensure you make the most of this energy."
            }
            default_response = f"Thank you for sharing. I understand you're primarily feeling **{primary_emotion.capitalize() if primary_emotion else 'a mix of things'}** regarding {career_stage}. Let's delve deeper into the specifics."
            st.info(empathetic_responses.get(primary_emotion, default_response)) # Use st.info for softer look

            st.markdown("---") # Separator within the card

            # Display analysis details using columns
            col1, col2 = st.columns(2)
            with col1:
                 if 'primary_emotion' in analysis:
                     st.metric("Primary Feeling", analysis['primary_emotion'].capitalize())
                 if 'career_stage' in analysis:
                      st.metric("Identified Career Stage", analysis['career_stage'].capitalize())
                 if 'intensity' in analysis:
                     intensity = analysis['intensity']
                     # Simple intensity display or use a progress bar like before
                     st.metric("Intensity Level", f"{intensity}/10")
                     # Optional: Progress bar visual
                     # st.progress(intensity / 10)

            with col2:
                 triggers = analysis.get('triggers', [])
                 if triggers and isinstance(triggers, list):
                     st.markdown("**Potential Triggers:**")
                     st.markdown(''.join(f"- {trigger}\n" for trigger in triggers))
                 else:
                     st.markdown("**Potential Triggers:** _None identified_")

                 growth_opps = analysis.get('growth_opportunities', [])
                 if growth_opps and isinstance(growth_opps, list):
                     st.markdown("**Identified Growth Opportunities:**")
                     st.markdown(''.join(f"- {opp}\n" for opp in growth_opps))
                 else:
                     st.markdown("**Identified Growth Opportunities:** _None identified_")

            # Implied Goals outside columns
            implied_goals = analysis.get('implied_goals', [])
            if implied_goals and isinstance(implied_goals, list):
                 st.markdown("---")
                 st.markdown("**Possible Underlying Goals:**")
                 st.markdown(''.join(f"- {goal}\n" for goal in implied_goals))

            st.markdown('</div>', unsafe_allow_html=True) # Close card-container


    # --- Generate/Display Growth Plan ---
    st.markdown("### 🚀 Your Personalized Growth Plan")
    plan_placeholder = st.empty()
    growth_plan = get_growth_plan(entry_id) # Fetch from DB

    if growth_plan and 'error' not in growth_plan: # Plan exists and is valid
        with plan_placeholder.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader(growth_plan.get('title', 'Growth Plan'))
            if vision := growth_plan.get('vision_statement'):
                 st.markdown(f"**Vision:** *{vision}*")

            st.markdown("**Action Steps:**")
            steps = growth_plan.get('steps', [])
            if steps and isinstance(steps, list):
                for i, step in enumerate(steps):
                     with st.expander(f"**Step {i+1}: {step.get('title', 'Untitled Step')}** (Timeframe: {step.get('timeframe', 'N/A')})", expanded= (i==0) ): # Expand first step
                          st.markdown(step.get('description', 'No description.'))
                          indicators = step.get('success_indicators', [])
                          if indicators and isinstance(indicators, list):
                               st.markdown("**Success looks like:**")
                               st.markdown(''.join(f"- {ind}\n" for ind in indicators))
            else:
                 st.markdown("_No specific steps defined._")


            outcomes = growth_plan.get('expected_outcomes', [])
            if outcomes and isinstance(outcomes, list):
                 st.markdown("**Expected Outcomes:**")
                 st.markdown(''.join(f"- {out}\n" for out in outcomes))

            obstacles = growth_plan.get('potential_obstacles', [])
            if obstacles and isinstance(obstacles, list):
                 st.markdown("**Potential Obstacles:**")
                 st.markdown(''.join(f"- {obs}\n" for obs in obstacles))

            if tip := growth_plan.get('accountability_tip'):
                 st.markdown(f"**Accountability Tip:** {tip}")

            st.markdown('</div>', unsafe_allow_html=True) # Close card

    elif analysis and 'error' not in analysis: # Analysis exists, but plan doesn't
        with plan_placeholder.container():
            st.info("A personalized growth plan can help structure your next steps.")
            if st.button("🌱 Generate Growth Plan", key="generate_plan_aishura", type="primary"):
                 with st.spinner("Creating your growth plan..."):
                      render_loading_animation("Creating your growth plan...")
                      new_plan = generate_growth_plan(analysis) # Pass the valid analysis
                      if isinstance(new_plan, dict) and 'error' not in new_plan:
                           if save_growth_plan(entry_id, new_plan):
                                st.success("Growth plan generated successfully!")
                                st.rerun() # Rerun to display the new plan
                           else:
                                st.error("Failed to save the generated growth plan.")
                      else:
                           error_msg = new_plan.get('error', 'Unknown error') if isinstance(new_plan, dict) else "Unknown error"
                           st.error(f"Could not generate growth plan: {error_msg}")
                           if isinstance(new_plan, dict) and 'raw_response' in new_plan:
                                st.expander("Show Raw Debug Info").code(new_plan['raw_response'])
    else:
         # Cannot generate plan if analysis failed
         with plan_placeholder.container():
              st.warning("Growth plan cannot be generated until the analysis is successful.")


    # --- Generate/Display Action Suggestions ---
    st.markdown("### ✨ Actionable Suggestions")
    action_placeholder = st.empty()
    action_suggestions = get_action_suggestions(entry_id) # Fetch from DB

    if action_suggestions and 'error' not in action_suggestions: # Suggestions exist and are valid
        with action_placeholder.container():
             st.markdown('<div class="card-container">', unsafe_allow_html=True)
             if quote := action_suggestions.get('motivation_quote'):
                 st.markdown(f"*{quote}*")
                 st.markdown("---")

             immediate = action_suggestions.get('immediate_actions', [])
             if immediate and isinstance(immediate, list):
                  st.markdown("**⚡ Take Action Now:**")
                  st.markdown(''.join(f"- {act}\n" for act in immediate))

             prep = action_suggestions.get('preparation_guidance', [])
             if prep and isinstance(prep, list):
                  st.markdown("**🔧 Prepare For Success:**")
                  for item in prep:
                      if isinstance(item, dict) and 'item' in item and 'guidance' in item:
                           st.markdown(f"- **{item['item']}:** {item['guidance']}")

             skills = action_suggestions.get('skill_development', [])
             if skills and isinstance(skills, list):
                  st.markdown("**🌱 Skill Development:**")
                  for skill_item in skills:
                       if isinstance(skill_item, dict):
                            skill = skill_item.get('skill', 'N/A')
                            resource = skill_item.get('resource', 'N/A')
                            why = skill_item.get('why_important', 'N/A')
                            st.markdown(f"- **{skill}:** Try '{resource}'. *Why? {why}*")

             if insight := action_suggestions.get('key_insight'):
                  st.markdown("**Key Insight:**")
                  st.success(f"💡 {insight}") # Use success box for insight


             st.markdown('</div>', unsafe_allow_html=True) # Close card

    elif analysis and 'error' not in analysis: # Analysis exists, but actions don't
        with action_placeholder.container():
             st.info("Get specific, actionable steps you can take based on your situation.")
             # Pass analysis and potentially existing plan to generator
             if st.button("💡 Generate Action Suggestions", key="generate_actions_aishura", type="primary"):
                 with st.spinner("Generating actionable suggestions..."):
                      render_loading_animation("Generating actionable suggestions...")
                      # Fetch the plan again in case it was just generated
                      current_plan = get_growth_plan(entry_id)
                      new_actions = generate_action_suggestions(analysis, current_plan)
                      if isinstance(new_actions, dict) and 'error' not in new_actions:
                           if save_action_suggestions(entry_id, new_actions):
                                st.success("Action suggestions generated successfully!")
                                st.rerun() # Rerun to display
                           else:
                                st.error("Failed to save the generated action suggestions.")
                      else:
                           error_msg = new_actions.get('error', 'Unknown error') if isinstance(new_actions, dict) else "Unknown error"
                           st.error(f"Could not generate action suggestions: {error_msg}")
                           if isinstance(new_actions, dict) and 'raw_response' in new_actions:
                                st.expander("Show Raw Debug Info").code(new_actions['raw_response'])
    else:
        # Cannot generate actions if analysis failed
         with action_placeholder.container():
              st.warning("Action suggestions cannot be generated until the analysis is successful.")


    # --- Generate/Display Industry Insights (Optional based on flag) ---
    if st.session_state.get('show_industry_insights', False):
        st.markdown("### 📈 Industry & Market Insights")
        insights_placeholder = st.empty()
        # NOTE: Industry insights are NOT saved to DB in this version. Generate on demand.
        # If saving were implemented, you'd check get_industry_insights(entry_id) here.

        if analysis and 'error' not in analysis: # Only generate if analysis is valid
            with insights_placeholder.container():
                if 'generated_insights' not in st.session_state or st.session_state.generated_insights.get('entry_id') != entry_id:
                     # Generate insights if not already generated for this entry in this session run
                    if st.button("🔍 Generate Industry Insights", key="generate_industry_insights_aishura"):
                        with st.spinner("Fetching relevant industry insights..."):
                             render_loading_animation("Fetching relevant industry insights...")
                             current_plan = get_growth_plan(entry_id) # Needed for context
                             new_insights = generate_industry_insights(analysis, current_plan)
                             if isinstance(new_insights, dict) and 'error' not in new_insights:
                                  st.success("Industry insights generated.")
                                  # Store in session state for this run to avoid re-generation on minor interactions
                                  st.session_state.generated_insights = {'entry_id': entry_id, 'data': new_insights}
                                  st.rerun()
                             else:
                                  error_msg = new_insights.get('error', 'Unknown error') if isinstance(new_insights, dict) else "Unknown error"
                                  st.error(f"Could not generate industry insights: {error_msg}")
                                  if isinstance(new_insights, dict) and 'raw_response' in new_insights:
                                       st.expander("Show Raw Debug Info").code(new_insights['raw_response'])
                elif 'generated_insights' in st.session_state and st.session_state.generated_insights.get('entry_id') == entry_id:
                     # Display previously generated insights stored in session state
                     insights_data = st.session_state.generated_insights['data']
                     st.markdown('<div class="card-container">', unsafe_allow_html=True)
                     industry = insights_data.get('industry_identified', 'General Career')
                     st.subheader(f"Insights for: {industry}")

                     trends = insights_data.get('key_trends', [])
                     if trends and isinstance(trends, list):
                          st.markdown("**Key Trends:**")
                          st.markdown(''.join(f"- {t}\n" for t in trends))

                     skills = insights_data.get('in_demand_skills', [])
                     if skills and isinstance(skills, list):
                          st.markdown("**In-Demand Skills:**")
                          st.markdown(''.join(f"- {s}\n" for s in skills))

                     challenges = insights_data.get('potential_challenges', [])
                     if challenges and isinstance(challenges, list):
                           st.markdown("**Potential Challenges:**")
                           st.markdown(''.join(f"- {c}\n" for c in challenges))

                     growth_areas = insights_data.get('growth_areas', [])
                     if growth_areas and isinstance(growth_areas, list):
                          st.markdown("**Growth Areas:**")
                          st.markdown(''.join(f"- {g}\n" for g in growth_areas))

                     if path_insight := insights_data.get('career_path_insights'):
                          st.markdown(f"**Career Path Note:** {path_insight}")

                     st.markdown('</div>', unsafe_allow_html=True) # Close card

        else:
            with insights_placeholder.container():
                 st.warning("Industry insights require a successful analysis first.")

    # --- Navigation ---
    st.markdown("---")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("⬅️ Back to Journal List", key="back_to_journal_aishura", use_container_width=True):
            st.session_state.current_view = "journal"
            if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
            if 'generated_insights' in st.session_state: del st.session_state['generated_insights'] # Clear generated insights
            st.rerun()
    with nav_col2:
         if st.button("🏠 Back to Dashboard", key="back_to_dash_aishura", use_container_width=True):
             st.session_state.current_view = "main"
             if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
             if 'generated_insights' in st.session_state: del st.session_state['generated_insights']
             st.rerun()


def render_progress_page():
    """Render the progress and insights page."""
    render_brand_header()
    st.title("📊 Progress & Insights")
    st.markdown("Track your career reflection journey, emotional trends, and growth over time.")

    user_id = st.session_state.get('current_user_id')
    if not user_id:
        st.error("User ID not found. Please log in again.")
        st.session_state.current_view = "login"
        st.rerun()
        return

    # --- Overall Statistics ---
    st.subheader("📈 Your Journey Statistics")
    total_user_entries = get_total_entries_count(user_id)
    total_user_insights = get_total_insights_count(user_id) # Assuming insight = analysis generated
    # Calculate number of growth plans created by the user
    conn = st.session_state.db_conn
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(gp.id) FROM growth_plans gp
        JOIN journal_entries j ON gp.entry_id = j.id
        WHERE j.user_id = ?
    """, (user_id,))
    plans_created = c.fetchone()[0] or 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reflections", total_user_entries)
    with col2:
        st.metric("Analyses Generated", total_user_insights)
    with col3:
        st.metric("Growth Plans Created", plans_created)

    # --- Visualizations ---
    st.markdown("---")
    st.subheader("📉 Trends Over Time")

    # Fetch data for visualizations
    entry_history = get_user_entries(user_id, limit=50) # Get more data for trends
    valid_analysis_count = sum(1 for entry in entry_history if entry.get('analysis') and 'error' not in entry['analysis'])

    if valid_analysis_count < 3:
         st.info("Keep reflecting on your career! More visualizations will appear here once you have at least 3 analyzed entries.")
    else:
        # Reuse visualizations from the dashboard, perhaps with more data points
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
             st.markdown("**Emotion Frequency Distribution**")
             emotions, counts = get_user_emotion_distribution(user_id, limit=10) # Show top 10
             if emotions and counts:
                 fig_pie = px.pie(names=emotions, values=counts, title="Primary Emotion Breakdown", hole=0.3, color_discrete_sequence=px.colors.sequential.Purples_r)
                 fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                 fig_pie.update_layout(showlegend=False, margin=dict(t=50, b=0, l=0, r=0), title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#333"))
                 st.plotly_chart(fig_pie, use_container_width=True)
             else:
                 st.info("No emotion data available yet.")

        with viz_col2:
             st.markdown("**Intensity Trend**")
             timestamps, intensities_trend = get_user_intensity_trend(user_id, limit=50) # Show last 50
             if timestamps and intensities_trend and len(timestamps) > 1:
                 df_trend = pd.DataFrame({'Timestamp': timestamps, 'Intensity': intensities_trend})
                 fig_line = px.line(df_trend, x='Timestamp', y='Intensity', title="Reflection Intensity Over Time", markers=True, line_shape='spline')
                 fig_line.update_layout(yaxis_range=[0, 10.5], xaxis_title="", yaxis_title="Intensity (1-10)", margin=dict(t=50, b=20, l=40, r=20), title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa', font=dict(color="#333"))
                 fig_line.update_traces(line=dict(color=APP_COLOR_SECONDARY))
                 st.plotly_chart(fig_line, use_container_width=True)
             else:
                 st.info("Need at least 2 entries with intensity data to show trend.")

    # --- List of Growth Plans (Optional) ---
    # You could add a section listing all created growth plans for easy access.
    # st.markdown("---")
    # st.subheader("📘 Your Growth Plans")
    # Fetch and display plans similar to how entries are listed in journal page...


def render_settings_page():
    """Render the settings page."""
    render_brand_header()
    st.title("⚙️ Settings")

    user_id = st.session_state.get('current_user_id')
    username = st.session_state.get('current_user', 'User')

    if not user_id:
        st.error("User ID not found. Please log in again.")
        st.session_state.current_view = "login"
        st.rerun()
        return

    # --- User Profile ---
    st.subheader("👤 User Profile")
    st.markdown(f"**Username:** {username}")
    # Fetch email if stored
    conn = st.session_state.db_conn
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    email_result = c.fetchone()
    email = email_result[0] if email_result and email_result[0] else "Not provided"
    st.markdown(f"**Email:** {email}")

    # Add options to change email or password if needed
    # st.button("Change Email") ...
    # st.button("Change Password") ...

    # --- Feature Flags / Preferences ---
    st.markdown("---")
    st.subheader("🔧 Preferences")

    # Example: Toggle for Industry Insights (if made configurable)
    show_insights = st.checkbox(
         "Show Industry Insights on Analysis Page",
         value=st.session_state.get('show_industry_insights', True),
         key="toggle_industry_insights"
    )
    if show_insights != st.session_state.get('show_industry_insights', True):
        st.session_state.show_industry_insights = show_insights
        st.rerun() # Rerun to reflect change immediately (optional)

    # Add more settings as needed (e.g., notification preferences, data export)

    # --- Account Management ---
    st.markdown("---")
    st.subheader("🔒 Account Management")

    if st.button("Logout", key="settings_logout", type="secondary"): # Use secondary style
        # Reuse logout logic from sidebar
        keys_to_clear = ['authenticated', 'current_user', 'current_user_id', 'current_view', 'current_entry_id', 'demo_mode', 'generated_insights']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.current_view = "login"
        st.success("Logged out successfully.")
        st.rerun()

    # Add Danger Zone for account deletion if required
    # with st.expander("🚨 Danger Zone"):
    #     st.warning("This action is irreversible!")
    #     if st.button("Delete My Account", type="primary"):
    #         # Implement account deletion logic here (requires confirmation)
    #         st.error("Account deletion not yet implemented.")


# --- MAIN APPLICATION LOGIC ---
def main():
    """Main function to control the application flow."""
    init_state() # Initialize session state on each run

    current_view = st.session_state.get('current_view', 'login') # Default to login

    # Render sidebar only if authenticated
    if st.session_state.get('authenticated', False):
         render_sidebar()

    # --- View Routing ---
    if current_view == "login":
        render_login_page()
    elif not st.session_state.get('authenticated', False):
         # If not authenticated and trying to access other pages, redirect to login
         st.session_state.current_view = "login"
         st.warning("Please log in to continue.")
         st.rerun()
    elif current_view == "main":
        render_main_dashboard()
    elif current_view == "journal":
        render_journal_page()
    elif current_view == "analysis":
        entry_id = st.session_state.get('current_entry_id')
        if entry_id:
             render_analysis_page(entry_id)
        else:
             st.warning("No reflection selected. Please choose one from the Journal or Dashboard.")
             st.session_state.current_view = "journal" # Redirect to journal
             st.rerun()
    elif current_view == "progress":
        render_progress_page()
    elif current_view == "settings":
        render_settings_page()
    else:
        # Fallback to dashboard if view is unknown
        st.warning(f"Unknown view: {current_view}. Redirecting to dashboard.")
        st.session_state.current_view = "main"
        st.rerun()


if __name__ == "__main__":
    main()
