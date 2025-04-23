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
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import requests
import base64

# --- CONSTANTS ---
APP_NAME = "Aishura"
APP_TAGLINE = "Your AI Career Coach & Growth Partner"
APP_VERSION = "1.0.0"
APP_COLOR_PRIMARY = "#6C63FF"  # Primary brand color (purple)
APP_COLOR_SECONDARY = "#FF6584"  # Secondary brand color (pink-red)
APP_LOGO_URL = "https://via.placeholder.com/150x150.png?text=Aishura"  # Placeholder logo

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
    }
    
    /* Metrics styling */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Card-like containers */
    .card-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Timeline styling for journal entries */
    .timeline-item {
        padding-left: 20px;
        border-left: 2px solid var(--primary);
        margin-bottom: 15px;
        padding-bottom: 15px;
    }
    
    /* Emotion tags styling */
    .emotion-tag {
        background-color: var(--primary);
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    
    /* For the dashboard stats */
    .stat-card {
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        background-color: white;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--dark);
        color: white;
    }
    
    /* Remove padding from containers */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Enhance header margins */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5em !important;
        margin-bottom: 0.5em !important;
    }
</style>
""", unsafe_allow_html=True)

# --- DATABASE SETUP ---
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
        entry_id TEXT NOT NULL,
        primary_emotion TEXT,
        intensity INTEGER,
        triggers TEXT,
        growth_opportunities TEXT,
        raw_data TEXT,
        FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
    )
    ''')
    
    # Create growth plans table
    c.execute('''
    CREATE TABLE IF NOT EXISTS growth_plans (
        id TEXT PRIMARY KEY,
        entry_id TEXT NOT NULL,
        title TEXT,
        steps TEXT,
        expected_outcomes TEXT,
        raw_data TEXT,
        FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
    )
    ''')
    
    # Create action suggestions table
    c.execute('''
    CREATE TABLE IF NOT EXISTS action_suggestions (
        id TEXT PRIMARY KEY,
        entry_id TEXT NOT NULL,
        immediate_actions TEXT,
        preparation_guidance TEXT,
        key_insight TEXT,
        raw_data TEXT,
        FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
    )
    ''')
    
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
    # Using the latest Gemini model for best results
    MODEL_NAME = "gemini-2.0-flash"
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"🔴 Error configuring Gemini API: {e}")
    st.stop()

# --- LLM PROMPTS ---
# Career Analysis Prompt (replaces previous emotion prompt)
CAREER_ANALYSIS_PROMPT = """SYSTEM: You are Aishura, an empathetic AI career assistant with expertise in career development, job search strategies, and professional growth. Analyze the user's entry for their primary career-related emotion, intensity (1-10), potential triggers, growth opportunities, and career goals. Respond ONLY with a valid JSON object. No explanations.

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

# Growth Plan Prompt (expanded from original)
GROWTH_PLAN_PROMPT = """SYSTEM: You are Aishura, an AI career coach with expertise in professional development. Create a concise, actionable growth plan based on the user's career concern analysis. The plan should be optimistic yet realistic, focused on tangible steps, and incorporate both short-term actions and long-term development. Respond ONLY with a valid JSON object. No explanations.

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

# Action Suggestions Prompt (expanded from original)
ACTION_SUGGESTION_PROMPT = """SYSTEM: You are Aishura, an AI career assistant providing proactive, actionable career advice. Your guidance should be specific, actionable, and personalized based on the user's current situation. Respond ONLY with a valid JSON object. No explanations.

USER: Based on the user's career concern analysis and growth plan, suggest concrete next steps. Provide practical, immediate actions they can take today.

User Profile:
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
      "resource": "string - specific resource type or approach for development",
      "why_important": "string - brief explanation of importance to their situation"
    }
  ],
  "key_insight": "string - single, concise, encouraging insight related to their situation",
  "motivation_quote": "string - brief motivational quote relevant to their career situation"
}
"""

# Industry-Specific Insights Prompt (new feature)
INDUSTRY_INSIGHTS_PROMPT = """SYSTEM: You are Aishura, an AI career assistant with expertise across various industries. Provide focused insights about the user's industry or target industry based on their career situation. Respond ONLY with a valid JSON object. No explanations.

USER: Based on the user's career information, provide industry-specific insights relevant to their situation:

User Profile: {profile_data_json}

JSON Structure: {
  "industry_identified": "string - the industry you've identified from the user's information",
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
        st.session_state.show_industry_insights = True
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    
    # Database connection
    if 'db_conn' not in st.session_state:
        st.session_state.db_conn = init_db()
    
    # Track metrics for demo
    if 'total_entries' not in st.session_state:
        st.session_state.total_entries = get_total_entries_count()
    if 'insights_generated' not in st.session_state:
        st.session_state.insights_generated = get_total_insights_count()

# --- DATA PERSISTENCE FUNCTIONS ---
def hash_password(password):
    """Create a secure hash of a password."""
    return hashlib.sha256(password.encode()).hexdigest()

def user_exists(username):
    """Check if a username exists in the database."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    return c.fetchone() is not None

def verify_user(username, password):
    """Verify user credentials."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    if user:
        # Update last login time
        c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user[0],))
        conn.commit()
        return user[0]  # Return user_id
    return None

def create_user(username, password, email=None):
    """Create a new user in the database."""
    if user_exists(username):
        return False, "Username already taken"
    
    try:
        conn = st.session_state.db_conn
        c = conn.cursor()
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(password)
        c.execute(
            "INSERT INTO users (id, username, password, email) VALUES (?, ?, ?, ?)",
            (user_id, username, hashed_password, email)
        )
        conn.commit()
        return True, user_id
    except Exception as e:
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
    except Exception as e:
        st.error(f"Error saving journal entry: {e}")
        return None

def save_analysis(entry_id, analysis_data):
    """Save analysis results to the database."""
    if not isinstance(analysis_data, dict):
        return False
    
    conn = st.session_state.db_conn
    c = conn.cursor()
    analysis_id = str(uuid.uuid4())
    
    try:
        # Convert lists to comma-separated strings for storage
        triggers = ','.join(analysis_data.get('triggers', [])) if isinstance(analysis_data.get('triggers'), list) else None
        growth_opportunities = ','.join(analysis_data.get('growth_opportunities', [])) if isinstance(analysis_data.get('growth_opportunities'), list) else None
        
        c.execute(
            """INSERT INTO analysis 
            (id, entry_id, primary_emotion, intensity, triggers, growth_opportunities, raw_data) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id, 
                entry_id,
                analysis_data.get('primary_emotion'),
                analysis_data.get('intensity'),
                triggers,
                growth_opportunities,
                json.dumps(analysis_data)
            )
        )
        conn.commit()
        # Update metrics
        st.session_state.insights_generated = get_total_insights_count()
        return True
    except Exception as e:
        st.error(f"Error saving analysis: {e}")
        return False

def save_growth_plan(entry_id, plan_data):
    """Save growth plan to the database."""
    if not isinstance(plan_data, dict):
        return False
    
    conn = st.session_state.db_conn
    c = conn.cursor()
    plan_id = str(uuid.uuid4())
    
    try:
        # Convert complex data to JSON strings for storage
        steps = json.dumps(plan_data.get('steps', []))
        outcomes = json.dumps(plan_data.get('expected_outcomes', []))
        
        c.execute(
            """INSERT INTO growth_plans 
            (id, entry_id, title, steps, expected_outcomes, raw_data) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                plan_id, 
                entry_id,
                plan_data.get('title'),
                steps,
                outcomes,
                json.dumps(plan_data)
            )
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving growth plan: {e}")
        return False

def save_action_suggestions(entry_id, action_data):
    """Save action suggestions to the database."""
    if not isinstance(action_data, dict):
        return False
    
    conn = st.session_state.db_conn
    c = conn.cursor()
    action_id = str(uuid.uuid4())
    
    try:
        # Convert complex data to JSON strings for storage
        immediate_actions = json.dumps(action_data.get('immediate_actions', []))
        preparation_guidance = json.dumps(action_data.get('preparation_guidance', []))
        
        c.execute(
            """INSERT INTO action_suggestions 
            (id, entry_id, immediate_actions, preparation_guidance, key_insight, raw_data) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                action_id, 
                entry_id,
                immediate_actions,
                preparation_guidance,
                action_data.get('key_insight'),
                json.dumps(action_data)
            )
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving action suggestions: {e}")
        return False

def get_user_entries(user_id, limit=10):
    """Get journal entries for a user with their analysis data."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    
    c.execute("""
        SELECT j.id, j.entry_text, j.timestamp, a.primary_emotion, a.intensity, a.raw_data
        FROM journal_entries j
        LEFT JOIN analysis a ON j.id = a.entry_id
        WHERE j.user_id = ?
        ORDER BY j.timestamp DESC
        LIMIT ?
    """, (user_id, limit))
    
    entries = []
    for row in c.fetchall():
        entry_id, entry_text, timestamp, primary_emotion, intensity, raw_analysis = row
        
        # Parse raw analysis if available
        analysis = None
        if raw_analysis:
            try:
                analysis = json.loads(raw_analysis)
            except:
                analysis = {
                    "primary_emotion": primary_emotion,
                    "intensity": intensity
                }
        
        entries.append({
            "id": entry_id,
            "journal_entry": entry_text,
            "timestamp": timestamp,
            "analysis": analysis
        })
    
    return entries

def get_journal_entry(entry_id):
    """Get a specific journal entry with its analysis."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    
    c.execute("""
        SELECT j.id, j.entry_text, j.timestamp, a.raw_data
        FROM journal_entries j
        LEFT JOIN analysis a ON j.id = a.entry_id
        WHERE j.id = ?
    """, (entry_id,))
    
    row = c.fetchone()
    if not row:
        return None
    
    entry_id, entry_text, timestamp, raw_analysis = row
    
    # Parse raw analysis if available
    analysis = None
    if raw_analysis:
        try:
            analysis = json.loads(raw_analysis)
        except:
            analysis = {}
    
    return {
        "id": entry_id,
        "journal_entry": entry_text,
        "timestamp": timestamp,
        "analysis": analysis
    }

def get_growth_plan(entry_id):
    """Get the growth plan for an entry."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    
    c.execute("SELECT raw_data FROM growth_plans WHERE entry_id = ?", (entry_id,))
    row = c.fetchone()
    
    if row and row[0]:
        try:
            return json.loads(row[0])
        except:
            return {"error": "Failed to parse growth plan data"}
    
    return None

def get_action_suggestions(entry_id):
    """Get the action suggestions for an entry."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    
    c.execute("SELECT raw_data FROM action_suggestions WHERE entry_id = ?", (entry_id,))
    row = c.fetchone()
    
    if row and row[0]:
        try:
            return json.loads(row[0])
        except:
            return {"error": "Failed to parse action suggestions data"}
    
    return None

def get_industry_insights(entry_id):
    """Placeholder for getting industry insights (not stored yet)."""
    # In a full implementation, this would fetch from a database
    # For MVP, we'll generate on-demand
    return None

def get_total_entries_count():
    """Get total number of entries for analytics."""
    try:
        conn = st.session_state.db_conn
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM journal_entries")
        result = c.fetchone()
        return result[0] if result else 0
    except:
        return 0

def get_total_insights_count():
    """Get total number of insights generated."""
    try:
        conn = st.session_state.db_conn
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM analysis")
        result = c.fetchone()
        return result[0] if result else 0
    except:
        return 0

def get_user_emotion_distribution(user_id, limit=20):
    """Get distribution of emotions for visualization."""
    conn = st.session_state.db_conn
    c = conn.cursor()
    
    c.execute("""
        SELECT a.primary_emotion, COUNT(*) as count
        FROM analysis a
        JOIN journal_entries j ON a.entry_id = j.id
        WHERE j.user_id = ? AND a.primary_emotion IS NOT NULL
        GROUP BY a.primary_emotion
        ORDER BY count DESC
        LIMIT ?
    """, (user_id, limit))
    
    emotions = []
    counts = []
    for emotion, count in c.fetchall():
        emotions.append(emotion)
        counts.append(count)
    
    return emotions, counts

# --- LLM INTERACTION LOGIC ---
def call_gemini_llm(prompt: str) -> Dict:
    """Call the Gemini LLM with error handling and JSON parsing."""
    response_data = {"raw_response": None, "error": None}
    
    try:
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        
        # Check for safety blocks before accessing text
        if not response.candidates:
            response_data["error"] = "Content blocked by safety filters or other reason."
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'unknown')
            st.error(response_data["error"] + f" Finish reason: {block_reason}")
            return response_data

        raw_text = response.text
        response_data["raw_response"] = raw_text

        try:
            # Try to parse the raw JSON response
            parsed_json = json.loads(raw_text)
            return parsed_json
        except json.JSONDecodeError:
            # If that fails, try to clean it up first
            cleaned_text = raw_text.strip()
            
            # Remove markdown code block formatting if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text.removeprefix("```json").removesuffix("```").strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.removeprefix("```").removesuffix("```").strip()
            
            try:
                parsed_json = json.loads(cleaned_text)
                return parsed_json
            except json.JSONDecodeError as json_err_clean:
                 response_data["error"] = f"Failed to parse JSON even after cleaning: {json_err_clean}"
                 st.warning(f"{response_data['error']}")
        except Exception as parse_err:
             response_data["error"] = f"Unexpected error processing LLM response content: {parse_err}"
             st.error(f"{response_data['error']}")

    except Exception as e:
        response_data["error"] = f"Gemini API call failed: {e}"
        st.error(f"{response_data['error']}")

    return response_data

def analyze_career_entry(journal_entry: str) -> Dict:
    """Analyze a career journal entry using the LLM."""
    prompt = CAREER_ANALYSIS_PROMPT.format(journal_entry_text=journal_entry)
    return call_gemini_llm(prompt)

def generate_growth_plan(emotion_analysis: Dict) -> Dict:
    """Generate a personalized growth plan."""
    if not emotion_analysis or not isinstance(emotion_analysis, dict):
        return {"error": "Cannot generate plan without valid analysis."}
    
    input_json_str = json.dumps(emotion_analysis, indent=2)
    prompt = GROWTH_PLAN_PROMPT.format(analysis_and_goals_json=input_json_str)
    return call_gemini_llm(prompt)

def generate_action_suggestions(emotion_analysis: Dict, growth_plan: Optional[Dict] = None) -> Dict:
    """Generate actionable career suggestions."""
    if not emotion_analysis or not isinstance(emotion_analysis, dict):
        return {"error": "Cannot generate actions without valid analysis."}
    
    profile_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan or {"message": "No growth plan available."},
    }
    
    input_json_str = json.dumps(profile_data, indent=2)
    prompt = ACTION_SUGGESTION_PROMPT.format(profile_data_json=input_json_str)
    return call_gemini_llm(prompt)

def generate_industry_insights(emotion_analysis: Dict, growth_plan: Optional[Dict] = None) -> Dict:
    """Generate industry-specific insights."""
    if not emotion_analysis or not isinstance(emotion_analysis, dict):
        return {"error": "Cannot generate industry insights without valid analysis."}
    
    profile_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan or {"message": "No growth plan available."},
    }
    
    input_json_str = json.dumps(profile_data, indent=2)
    prompt = INDUSTRY_INSIGHTS_PROMPT.format(profile_data_json=input_json_str)
    return call_gemini_llm(prompt)

# --- UI COMPONENTS ---
def render_brand_header():
    """Render the brand header/logo area."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(APP_LOGO_URL, width=80)
    with col2:
        st.markdown(f"<h1 class='main-header'>{APP_NAME}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p>{APP_TAGLINE}</p>", unsafe_allow_html=True)

def render_loading_animation():
    """Render a custom loading animation for AI processing."""
    # This is a placeholder - in a real app, use a branded animation
    loader_html = """
    <div style="display:flex;justify-content:center;margin:2rem 0;">
        <div style="border:8px solid #f3f3f3;border-top:8px solid #6C63FF;border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite"></div>
    </div>
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """
    st.markdown(loader_html, unsafe_allow_html=True)

def render_login_page():
    """Render the login/signup page."""
    # Brand header
    render_brand_header()
    
    # Hero section
    st.markdown("""
    <div style="text-align:center;padding:20px 0;">
        <h2>Your AI-Powered Career Growth Partner</h2>
        <p>Unlock your professional potential with personalized insights and actionable guidance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card-container" style="text-align:center;">
            <h3>✨ Smart Career Analysis</h3>
            <p>AI-powered insights to understand your current situation and identify growth opportunities</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card-container" style="text-align:center;">
            <h3>🚀 Growth Planning</h3>
            <p>Personalized roadmaps to help you achieve your professional goals step-by-step</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card-container" style="text-align:center;">
            <h3>💡 Actionable Guidance</h3>
            <p>Practical suggestions and industry insights to accelerate your career progress</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Login/Signup columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("💼 Login")
        with st.form("login_form_aishura"):
            uname = st.text_input("Username", key="login_u_aishura")
            pwd = st.text_input("Password", type="password", key="login_p_aishura")
            login_col1, login_col2 = st.columns([3, 1])
            with login_col1:
                login_btn = st.form_submit_button("Login", use_container_width=True)
            with login_col2:
                demo_btn = st.form_submit_button("Demo Mode", use_container_width=True)
            
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
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
            
            if demo_btn:
                st.session_state.current_user = "Demo_User"
                st.session_state.current_user_id = "demo-id-123"
                st.session_state.authenticated = True
                st.session_state.current_view = "main"
                st.session_state.demo_mode = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("✨ Sign Up")
        with st.form("signup_form_aishura"):
            s_uname = st.text_input("Choose Username", key="signup_u_aishura")
            s_email = st.text_input("Email (optional)", key="signup_e_aishura")
            s_pwd = st.text_input("Choose Password", type="password", key="signup_p_aishura")
            s_pwd_confirm = st.text_input("Confirm Password", type="password", key="signup_pc_aishura")
            signup_btn = st.form_submit_button("Create Account", use_container_width=True)
            
            if signup_btn:
                if not s_uname or not s_pwd:
                    st.warning("Username and password are required.")
                elif s_pwd != s_pwd_confirm:
                    st.error("Passwords don't match.")
                elif len(s_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, result = create_user(s_uname, s_pwd, s_email)
                    if success:
                        st.success("Account created! Please log in.")
                    else:
                        st.error(f"Failed to create account: {result}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:30px;font-size:0.8rem;">
        <p>© 2025 Aishura AI - Your Career Growth Partner</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the navigation sidebar."""
    if not st.session_state.get('authenticated'):
        return
    
    with st.sidebar:
        st.image(APP_LOGO_URL, width=100)
        st.title(f"{APP_NAME}")
        
        user_name = st.session_state.current_user
        demo_badge = " 🔍 DEMO MODE" if st.session_state.demo_mode else ""
        st.markdown(f"### 👤 Welcome, {user_name}{demo_badge}")
        
        st.markdown("---")
        
        # Navigation menu
        st.subheader("Navigation")
        pages = {
            "🏠 Dashboard": "main",
            "📝 Career Journal": "journal", 
            "📊 Progress & Insights": "progress"
        }
        
        current_view = st.session_state.get('current_view', 'main')
        for page_name, view_id in pages.items():
            if st.sidebar.button(
                page_name, 
                key=f"nav_{view_id}",
                use_container_width=True,
                type="primary" if current_view == view_id else "secondary"
            ):
                st.session_state.current_view = view_id
                if view_id in ["main", "journal", "progress"]:
                    if 'current_entry_id' in st.session_state:
                        del st.session_state['current_entry_id']
                st.rerun()
        
        st.markdown("---")
        
        # Account/Settings section
        st.markdown("### Account")
        if st.sidebar.button("⚙️ Settings", key="settings_btn", use_container_width=True):
            st.session_state.current_view = "settings"
            st.rerun()
            
        if st.sidebar.button("🔒 Logout", key="logout_aishura", use_container_width=True):
            for key in ['authenticated', 'current_user', 'current_user_id', 'current_view', 'current_entry_id', 'demo_mode']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_view = "login"
            st.rerun()
        
        # Demo info
        if st.session_state.demo_mode:
            st.markdown("---")
            st.markdown("### Demo Information")
            st.info("This is a demonstration version with simulated data. Some features may use pre-generated content.")
        
        # Version info
        st.markdown("---")
        st.caption(f"Version {APP_VERSION}")

def render_main_dashboard():
    """Render the main dashboard view."""
    st.title("✨ Your Career Dashboard")
    
    user_id = st.session_state.current_user_id
    entry_history = get_user_entries(user_id)
    
    # Welcome message and quick action
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### How can I help you today, {st.session_state.current_user}?")
    with col2:
        if st.button("📝 Share Career Thoughts", type="primary", use_container_width=True, key="new_entry_dash_aishura"):
            st.session_state.current_view = "journal"
            if 'current_entry_id' in st.session_state:
                del st.session_state['current_entry_id']
            st.rerun()
    
    # Stats overview
    st.subheader("Your Progress Overview")
    stat1, stat2, stat3, stat4 = st.columns(4)
    
    with stat1:
        st.markdown("""
        <div class="stat-card">
            <h2 style="color:#6C63FF;margin:0;">{}</h2>
            <p style="margin:0;">Career Reflections</p>
        </div>
        """.format(len(entry_history)), unsafe_allow_html=True)
    
    with stat2:
        # Calculate top emotion if we have entries
        top_emotion = "N/A"
        if entry_history:
            emotions = {}
            for entry in entry_history:
                if entry.get('analysis') and entry['analysis'].get('primary_emotion'):
                    emotion = entry['analysis']['primary_emotion']
                    emotions[emotion] = emotions.get(emotion, 0) + 1
            
            if emotions:
                top_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        st.markdown("""
        <div class="stat-card">
            <h2 style="color:#FF6584;margin:0;">{}</h2>
            <p style="margin:0;">Common Theme</p>
        </div>
        """.format(top_emotion), unsafe_allow_html=True)
    
    with stat3:
        # Calculate average intensity if we have entries
        avg_intensity = "N/A"
        intensities = []
        for entry in entry_history:
            if entry.get('analysis') and entry['analysis'].get('intensity'):
                intensities.append(entry['analysis']['intensity'])
        
        if intensities:
            avg_intensity = f"{sum(intensities) / len(intensities):.1f}/10"
        
        st.markdown("""
        <div class="stat-card">
            <h2 style="color:#6C63FF;margin:0;">{}</h2>
            <p style="margin:0;">Avg. Intensity</p>
        </div>
        """.format(avg_intensity), unsafe_allow_html=True)
    
    with stat4:
        # Count growth plans
        plans_count = 0
        for entry in entry_history:
            plan = get_growth_plan(entry['id'])
            if plan and 'error' not in plan:
                plans_count += 1
        
        st.markdown("""
        <div class="stat-card">
            <h2 style="color:#FF6584;margin:0;">{}</h2>
            <p style="margin:0;">Growth Plans</p>
        </div>
        """.format(plans_count), unsafe_allow_html=True)
    
    # Recent entries section
    st.markdown("---")
    st.subheader("Recent Career Insights")
    
    if not entry_history:
        st.info("📝 Click the button above to share how you're feeling about your career or what challenges you're facing.")
    else:
        # Display the most recent 3 entries
        for i, entry in enumerate(entry_history[:3]):
            entry_id = entry.get('id')
            ts_str = entry.get('timestamp', '')
            display_time = ts_str[:16].replace('T', ' ') if ts_str else '?'
            
            analysis = entry.get('analysis', {})
            is_analysis_ok = isinstance(analysis, dict) and 'error' not in analysis
            p_emotion = analysis.get('primary_emotion', 'Entry') if is_analysis_ok else 'Entry'
            
            # Create a card-like container for each entry
            st.markdown(f"""
            <div class="card-container">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span class="emotion-tag">{p_emotion}</span>
                        <span style="color:#666;font-size:0.9rem;"> · {display_time}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Content summary
            st.markdown(f"<div style='padding:0 10px;'>", unsafe_allow_html=True)
            st.caption("Your thoughts:")
            st.write(entry.get('journal_entry', 'N/A')[:150] + "..." if len(entry.get('journal_entry', 'N/A')) > 150 else entry.get('journal_entry', 'N/A'))
            
            # Analysis summary (if available)
            if is_analysis_ok:
                if analysis.get('triggers'):
                    st.caption("Key triggers:")
                    st.write(", ".join(analysis['triggers'][:2]) + ("..." if len(analysis['triggers']) > 2 else ""))
                
                if analysis.get('growth_opportunities'):
                    st.caption("Growth areas:")
                    st.write(", ".join(analysis['growth_opportunities'][:2]) + ("..." if len(analysis['growth_opportunities']) > 2 else ""))
            
            # View details button
            if st.button("View Full Analysis & Actions", key=f"view_dash_aishura_{entry_id}", use_container_width=True):
                st.session_state.current_entry_id = entry_id
                st.session_state.current_view = "analysis"
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Don't add margin after the last item
            if i < len(entry_history[:3]) - 1:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Visualization section (if user has enough data)
    if len(entry_history) >= 3:
        st.markdown("---")
        st.subheader("Your Career Emotion Trends")
        
        emotions, counts = get_user_emotion_distribution(user_id)
        
        if emotions and counts:
            # Create a Plotly bar chart
            fig = px.bar(
                x=emotions, 
                y=counts,
                labels={'x': 'Emotion', 'y': 'Frequency'},
                title="Your Most Common Career Emotions",
                color=counts,
                color_continuous_scale=px.colors.sequential.Purp
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="",
                yaxis_title="",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Continue sharing your career thoughts to build emotion trend insights.")
    
    # Call to action section
    st.markdown("---")
    cta_col1, cta_col2, cta_col3 = st.columns(3)
    
    with cta_col1:
        st.markdown("""
        <div class="card-container" style="text-align:center;">
            <h3>📝 Journal</h3>
            <p>Share your career thoughts and get AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Journaling", key="cta_journal", use_container_width=True):
            st.session_state.current_view = "journal"
            st.rerun()
    
    with cta_col2:
        st.markdown("""
        <div class="card-container" style="text-align:center;">
            <h3>📊 Progress</h3>
            <p>Track your career journey and growth over time</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Progress", key="cta_progress", use_container_width=True):
            st.session_state.current_view = "progress"
            st.rerun()
    
    with cta_col3:
        st.markdown("""
        <div class="card-container" style="text-align:center;">
            <h3>💼 Industry Insights</h3>
            <p>Get customized advice for your professional field</p>
        </div>
        """, unsafe_allow_html=True)
        
        # This would navigate to a specific feature
        if st.button("Explore Insights", key="cta_insights", use_container_width=True):
            # For MVP, just go to journal to create more entries
            st.session_state.current_view = "journal"
            st.rerun()

def render_journal_page():
    """Render the career journal entry page."""
    st.title("📝 Career Journal")
    st.subheader("How are you feeling about your career right now?")
    
    st.markdown("""
    <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:20px;">
        <p>Share your current career thoughts, concerns, aspirations, or goals. For example:</p>
        <ul>
            <li>How you're feeling about your current position</li>
            <li>Challenges you're facing in your job search</li>
            <li>Career transitions you're considering</li>
            <li>Growth opportunities you'd like to pursue</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Journal entry form
    with st.form("journal_form_aishura"):
        journal_entry = st.text_area(
            "Share your thoughts here:",
            height=180,
            key="journal_input_aishura",
            placeholder="What's on your mind regarding your career? (e.g., 'I'm feeling frustrated with my job search', 'I want to transition to a leadership role', 'I'm nervous about an upcoming interview')"
        )
        
        # Add some example buttons for demo purposes
        if st.session_state.demo_mode:
            st.caption("Demo examples (click to fill):")
            examples_col1, examples_col2, examples_col3 = st.columns(3)
            
            with examples_col1:
                if st.button("Job search example", key="example_1", use_container_width=True):
                    # Set example text in the text area
                    st.session_state.journal_input_aishura = "I've been applying to data scientist positions for three months now with no luck. I've had a few interviews but nothing has converted to an offer. I'm starting to doubt my skills and wonder if I need to learn more tools or languages. My resume seems strong but something's not working."
                    st.rerun()
            
            with examples_col2:
                if st.button("Leadership example", key="example_2", use_container_width=True):
                    st.session_state.journal_input_aishura = "I'm a senior developer and I want to move into a leadership position, but I'm finding it hard to break into management. My technical skills are strong, but I'm not sure how to demonstrate leadership potential. Should I speak to my manager about this or look for opportunities elsewhere?"
                    st.rerun()
            
            with examples_col3:
                if st.button("Career change example", key="example_3", use_container_width=True):
                    st.session_state.journal_input_aishura = "After 8 years in marketing, I'm considering switching to UX/UI design. I'm excited about the creative aspects but nervous about starting over. I've been taking online courses but I'm not sure if that's enough to make the transition. How do I leverage my existing skills while building new ones?"
                    st.rerun()
        
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col1:
            submit_btn = st.form_submit_button("🚀 Get Career Insights", type="primary", use_container_width=True)
        with submit_col2:
            clear_btn = st.form_submit_button("Clear", use_container_width=True)
        
        if clear_btn:
            st.session_state.journal_input_aishura = ""
            st.rerun()
        
        if submit_btn:
            if not journal_entry or len(journal_entry.strip()) < 10:
                st.warning("Please share a bit more about your career situation (at least 10 characters).")
            else:
                with st.spinner("🧠 Analyzing your career situation..."):
                    # For real implementation, show a branded loading animation
                    render_loading_animation()
                    
                    # Analyze the entry
                    analysis = analyze_career_entry(journal_entry.strip())
                    
                    # Save to database
                    if isinstance(analysis, dict) and 'error' not in analysis:
                        e_id = save_journal_entry(st.session_state.current_user_id, journal_entry.strip())
                        if e_id:
                            save_analysis(e_id, analysis)
                            st.session_state.current_entry_id = e_id
                            st.session_state.current_view = "analysis"
                            st.rerun()
                        else:
                            st.error("Failed to save your journal entry. Please try again.")
                    else:
                        st.error(f"Analysis failed: {analysis.get('error', 'Unknown error')}")
    
    # Past entries section
    st.markdown("---")
    st.subheader("Your Past Entries")
    
    user_id = st.session_state.current_user_id
    history = get_user_entries(user_id)
    
    if not history:
        st.info("You haven't created any journal entries yet.")
    else:
        # Create a timeline-style list of past entries
        for i, entry in enumerate(history):
            entry_id = entry.get('id')
            ts_str = entry.get('timestamp', '')
            display_time = ts_str[:16].replace('T', ' ') if ts_str else '?'
            
            analysis = entry.get('analysis', {})
            is_analysis_ok = isinstance(analysis, dict) and 'error' not in analysis
            p_emotion = analysis.get('primary_emotion', 'Entry') if is_analysis_ok else 'Entry'
            
            # Timeline item container
            st.markdown(f"""
            <div class="timeline-item">
                <p style="margin:0;">
                    <strong style="color:#6C63FF;">{display_time}</strong>
                    <span class="emotion-tag">{p_emotion}</span>
                </p>
                <p style="margin-top:5px;">{entry.get('journal_entry', '')[:100]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Button for viewing details
            btn_col1, btn_col2 = st.columns([4, 1])
            with btn_col2:
                if st.button("View", key=f"list_journ_aishura_{entry_id}", use_container_width=True):
                    st.session_state.current_entry_id = entry_id
                    st.session_state.current_view = "analysis"
                    st.rerun()

def render_analysis_page(entry_id):
    """Render the analysis page for a specific journal entry."""
    st.title("💬 Career Insight Analysis")
    
    user_id = st.session_state.current_user_id
    entry_data = get_journal_entry(entry_id)
    
    if not entry_data:
        st.error("Entry not found.")
        if st.button("Back to Journal", key="back_analysis_err_aishura"):
            st.session_state.current_view = "journal"
            st.rerun()
        return
    
    # Display the original entry with nice formatting
    st.markdown("""
    <div class="card-container">
        <h3>Your Career Input</h3>
        <blockquote style="border-left:3px solid #6C63FF;padding-left:15px;font-style:italic;">
            {entry_text}
        </blockquote>
        <p style="text-align:right;font-size:0.8rem;color:#666;">
            Shared on {timestamp}
        </p>
    </div>
    """.format(
        entry_text=entry_data.get('journal_entry', 'N/A'),
        timestamp=entry_data.get('timestamp', '')[:19].replace('T', ' ')
    ), unsafe_allow_html=True)
    
    # Get analysis data
    analysis = entry_data.get('analysis', {})
    analysis_successful = isinstance(analysis, dict) and 'error' not in analysis and 'raw_response' not in analysis
    
    # Empathetic Response
    st.markdown("### Aishura's Response")
    
    if analysis_successful and analysis.get('primary_emotion'):
        primary_emotion = analysis.get('primary_emotion', '').lower()
        career_stage = analysis.get('career_stage', 'your career').lower()
        
        # Create a more detailed, empathetic response
        empathetic_responses = {
            'anxious': f"I understand you're feeling **anxious** about {career_stage}. This is a common emotion when facing uncertainty. Let's work together to bring more clarity and confidence to your situation.",
            'frustrated': f"I see you're feeling **frustrated** with aspects of {career_stage}. This is completely valid - career challenges can be tough. Let's explore ways to overcome these obstacles together.",
            'excited': f"It's great to see you're feeling **excited** about {career_stage}! This positive energy is valuable as you move forward. Let's channel it into productive next steps.",
            'confused': f"I notice you're feeling **confused** about your direction in {career_stage}. Many professionals experience this. Let's bring some clarity to your situation and options.",
            'overwhelmed': f"I can see you're feeling **overwhelmed** with {career_stage} responsibilities or decisions. This is understandable. Let's break things down into manageable steps.",
            'confident': f"I'm glad to see you're feeling **confident** in your {career_stage} journey. This is a great foundation to build upon as you continue to grow professionally.",
            'hopeful': f"I notice a sense of **hope** in your thoughts about {career_stage}. That's wonderful - hope is a powerful motivator. Let's develop clear actions to move toward your vision.",
            'discouraged': f"I understand you're feeling **discouraged** about aspects of {career_stage}. Many professionals go through similar phases. Let's find ways to rebuild momentum and motivation.",
            'uncertain': f"I can see you're feeling **uncertain** about your path in {career_stage}. This is a normal part of professional growth. Let's explore ways to gain more clarity.",
            'ambitious': f"I notice your **ambition** regarding {career_stage}. That drive can be a powerful force for your growth. Let's channel it into strategic steps."
        }
        
        default_response = f"I understand you're feeling **{primary_emotion}** about {career_stage}. Many professionals experience similar emotions. Let's work together to navigate this situation effectively."
        
        empathetic_response = empathetic_responses.get(primary_emotion.lower(), default_response)
        
        st.success(empathetic_response)
    else:
        st.info("Let's break down your situation and find a path forward for your career growth.")
    
    # Display Analysis Details in a card-like container
    st.markdown("""
    <div class="card-container">
        <h3>Career Situation Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not analysis:
        st.warning("Analysis data is missing or incomplete.")
    elif 'error' in analysis:
        st.error(f"Analysis Error: {analysis['error']}")
        if 'raw_response' in analysis:
            st.code(analysis['raw_response'], language='text')
    elif 'raw_response' in analysis:
        st.warning("Analysis needs review:")
        st.code(analysis['raw_response'], language='text')
    else:
        # Display successful analysis with a better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotion and intensity
            if 'primary_emotion' in analysis:
                st.metric("Primary Feeling", analysis['primary_emotion'].capitalize())
            
            if 'intensity' in analysis:
                intensity = analysis['intensity']
                intensity_color = "#28a745" if intensity < 5 else ("#ffc107" if intensity < 8 else "#dc3545")
                
                st.markdown(f"""
                <div style="margin-bottom:20px;">
                    <p style="margin-bottom:5px;"><strong>Intensity</strong></p>
                    <div style="background-color:#e9ecef;height:20px;border-radius:10px;overflow:hidden;">
                        <div style="width:{intensity*10}%;background-color:{intensity_color};height:100%;"></div>
                    </div>
                    <p style="text-align:right;margin-top:5px;">{intensity}/10</p>
                </div>
                """, unsafe_allow_html=True)
            
            if 'career_stage' in analysis:
                st.metric("Career Stage", analysis['career_stage'].capitalize())
        
        with col2:
            # Triggers and Growth Opportunities
            triggers = analysis.get('triggers', [])
            if triggers:
                st.markdown("**Potential Triggers:**")
                for trigger in triggers:
                    st.markdown(f"- {trigger}")
            else:
                st.markdown("**Potential Triggers:** _None identified_")
            
            growth_opps = analysis.get('growth_opportunities', [])
            if growth_opps:
                st.markdown("**Growth Opportunities:**")
                for opp in growth_opps:
                    st.markdown(f"- {opp}")
            else:
                st.markdown("**Growth Opportunities:** _None identified_")
        
        # Implied Goals (if available)
        if 'implied_goals' in analysis and analysis['implied_goals']:
            st.markdown("**Career Goals Identified:**")
            for goal in analysis['implied_goals']:
                st.markdown(f"- {goal}")
    
    # Next Steps section
    st.markdown("---")
    st.subheader("Recommended Next Steps")
    
    col1, col2 = st.columns(2)
    
    # Check if we have a growth plan
    growth_plan = get_growth_plan(entry_id)
    plan_exists = isinstance(growth_plan, dict) and 'error' not in growth_plan
    
    # Check if we have action suggestions
    action_suggestions = get_action_suggestions(entry_id)
    actions_exist = isinstance(action_suggestions, dict) and 'error' not in action_suggestions
