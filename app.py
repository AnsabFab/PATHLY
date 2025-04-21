import os
import json
import datetime
import uuid
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any

# --- Configuration ---
# Load environment variables (like API keys) from a .env file
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("🔴 Error: GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop() # Stop execution if key is missing

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Note: Using gemini-1.5-flash-latest as the current high-speed model.
    MODEL_NAME = "gemini-1.5-flash-latest"
    # Safety settings can be adjusted if needed
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(MODEL_NAME)
    st.info(f"✅ Gemini API Configured (Model: {MODEL_NAME})")
except Exception as e:
    st.error(f"🔴 Error configuring Gemini API: {e}")
    st.stop()

# --- LLM Prompts (Focused for JSON Output) ---

EMOTION_PROMPT = """SYSTEM: You are an empathetic AI analyzing a journal entry. Respond ONLY with a valid JSON object containing the analysis. Do not include any introductory text, markdown formatting, or explanations outside the JSON structure.
USER: Analyze the following journal entry. Identify the primary emotion, its intensity (1-10), potential triggers (list of strings), and suggest growth opportunities (list of strings).
JSON Structure: {"primary_emotion": "string", "intensity": integer, "triggers": ["string"], "growth_opportunities": ["string"]}

Journal Entry:
{journal_entry_text}
"""

GROWTH_PLAN_PROMPT = """SYSTEM: You are an AI coach creating a growth plan. Respond ONLY with a valid JSON object. Do not include any introductory text, markdown formatting, or explanations outside the JSON structure.
USER: Based on the provided emotional analysis and optional goals, create a concise growth plan. It needs a title, 3-5 steps (each with title and description), and 2-3 expected outcomes (list of strings).
JSON Structure: {"title": "string", "steps": [{"title": "string", "description": "string"}], "expected_outcomes": ["string"]}

Analysis & Goals:
{analysis_and_goals_json}
"""

RESOURCE_PROMPT = """SYSTEM: You are an AI resource curator. Respond ONLY with a valid JSON object. Do not include any introductory text, markdown formatting, or explanations outside the JSON structure.
USER: Based on the user's emotional analysis and growth plan, suggest resources: key insights (list of strings), practical exercises (list of strings), and 2-3 relevant book/article titles (fictional or generic) with brief descriptions.
JSON Structure: {"key_insights": ["string"], "practical_exercises": ["string"], "recommended_readings": [{"title": "string", "description": "string"}]}

User Profile (Analysis & Plan):
{profile_data_json}
"""

# --- Session State Management ---
def init_state():
    """Initializes session state dictionaries for user data and app state."""
    if 'user_db' not in st.session_state: st.session_state.user_db = {}
    if 'emotion_db' not in st.session_state: st.session_state.emotion_db = {}
    if 'growth_plans_db' not in st.session_state: st.session_state.growth_plans_db = {}
    if 'resource_db' not in st.session_state: st.session_state.resource_db = {}

    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'current_user' not in st.session_state: st.session_state.current_user = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "login" if not st.session_state.authenticated else "main"
    if 'current_emotion_id' not in st.session_state: st.session_state.current_emotion_id = None

# --- Data Persistence Functions (In-Memory for MVP) ---
def save_user_data(user_id: str, data: Dict):
    st.session_state.user_db[user_id] = data

def save_emotion_entry(user_id: str, emotion_data: Dict):
    if user_id not in st.session_state.emotion_db: st.session_state.emotion_db[user_id] = []
    emotion_data['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat() # Use UTC
    emotion_data['id'] = str(uuid.uuid4())
    st.session_state.emotion_db[user_id].append(emotion_data)
    return emotion_data['id']

def get_user_emotion_history(user_id: str) -> List[Dict]:
    return st.session_state.emotion_db.get(user_id, [])

def get_emotion_entry(user_id: str, emotion_id: str) -> Optional[Dict]:
    for entry in get_user_emotion_history(user_id):
        if entry.get('id') == emotion_id: return entry
    return None

def save_growth_plan(user_id: str, emotion_id: str, plan_data: Dict):
    if user_id not in st.session_state.growth_plans_db: st.session_state.growth_plans_db[user_id] = {}
    st.session_state.growth_plans_db[user_id][emotion_id] = plan_data

def get_growth_plan(user_id: str, emotion_id: str) -> Optional[Dict]:
    return st.session_state.growth_plans_db.get(user_id, {}).get(emotion_id)

def save_resource(user_id: str, emotion_id: str, resource_data: Dict):
    if user_id not in st.session_state.resource_db: st.session_state.resource_db[user_id] = {}
    st.session_state.resource_db[user_id][emotion_id] = resource_data

def get_resources(user_id: str, emotion_id: str) -> Optional[Dict]:
    return st.session_state.resource_db.get(user_id, {}).get(emotion_id)

# --- LLM Interaction Logic (Using Gemini API) ---
def call_gemini_llm(prompt: str) -> Dict:
    """Calls the configured Gemini LLM and attempts to parse JSON response."""
    response_data = {"raw_response": None, "error": None}
    try:
        # Generate content using the Gemini API
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        raw_text = response.text
        response_data["raw_response"] = raw_text

        # Attempt to parse JSON strictly from the raw text
        try:
            # Simple parsing, assuming the model adhered to the JSON-only instruction
            parsed_json = json.loads(raw_text)
            return parsed_json # Return successful JSON directly
        except json.JSONDecodeError as json_err:
            # Handle cases where the output wasn't perfect JSON
            # Try cleaning common issues like markdown code blocks
            cleaned_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                parsed_json = json.loads(cleaned_text)
                return parsed_json # Return cleaned JSON
            except json.JSONDecodeError:
                 response_data["error"] = f"Failed to parse JSON from LLM response: {json_err}"
                 st.warning(f"{response_data['error']} Raw: {raw_text}")
        except Exception as parse_err:
             response_data["error"] = f"Unexpected error processing LLM response content: {parse_err}"
             st.error(f"{response_data['error']} Raw: {raw_text}")

    except genai.types.BlockedPromptError as bpe:
         response_data["error"] = f"Prompt blocked by safety settings: {bpe}"
         st.error(response_data["error"])
    except Exception as e:
        # Catch other potential API errors
        response_data["error"] = f"Gemini API call failed: {e}"
        st.error(f"{response_data['error']}")

    # Return dict with error or raw response if JSON failed
    return response_data

def analyze_emotion(journal_entry: str) -> Dict:
    """Analyzes emotion in a journal entry using the LLM."""
    prompt = EMOTION_PROMPT.format(journal_entry_text=journal_entry)
    return call_gemini_llm(prompt)

def generate_growth_plan(emotion_analysis: Dict, user_goals: Optional[Dict] = None) -> Dict:
    """Generates a growth plan based on emotion analysis and goals."""
    input_data = {
        "emotion_analysis": emotion_analysis,
        "user_goals": user_goals or {"general_goal": "Improve emotional well-being"}
    }
    input_json_str = json.dumps(input_data, indent=2)
    prompt = GROWTH_PLAN_PROMPT.format(analysis_and_goals_json=input_json_str)
    return call_gemini_llm(prompt)

def generate_resources_from_profile(emotion_analysis: Dict, growth_plan: Optional[Dict]) -> Dict:
    """Generates resource suggestions using LLM based on analysis and plan."""
    if not emotion_analysis:
        return {"error": "Cannot generate resources without emotion analysis."}
    profile_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan or {"message": "No growth plan available."},
    }
    input_json_str = json.dumps(profile_data, indent=2)
    prompt = RESOURCE_PROMPT.format(profile_data_json=input_json_str)
    st.info("Generating personalized resource suggestions with Gemini...")
    return call_gemini_llm(prompt)

# --- UI Rendering Functions (Similar to previous MVP) ---

def render_login_page():
    """Displays the login and sign-up forms."""
    st.title("🌱 Pathly MVP (Gemini)")
    st.subheader("Transform Emotional Experiences into Personal Growth")
    st.markdown("*(Investor Demo - AI Powered Insights)*")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username_gem")
            login_password = st.text_input("Password", type="password", key="login_password_gem")
            login_submitted = st.form_submit_button("Login")

            if login_submitted:
                # Simple auth for demo
                if login_username in st.session_state.user_db and st.session_state.user_db[login_username].get('password') == login_password:
                    st.session_state.current_user = login_username
                    st.session_state.authenticated = True
                    st.session_state.current_view = "main"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with col2:
        st.subheader("Sign Up")
        with st.form("signup_form"):
            signup_username = st.text_input("Choose Username", key="signup_username_gem")
            signup_password = st.text_input("Choose Password", type="password", key="signup_password_gem")
            # Basic validation for demo
            if st.form_submit_button("Sign Up"):
                if not signup_username or not signup_password:
                    st.warning("Username and password required.")
                elif signup_username in st.session_state.user_db:
                    st.error("Username already taken.")
                elif len(signup_password) < 4:
                     st.error("Password must be at least 4 characters.")
                else:
                    # Store credentials (in real app, hash password)
                    st.session_state.user_db[signup_username] = {'password': signup_password}
                    st.success("Account created! Please login.")
                    st.balloons()


def render_sidebar():
    """Renders sidebar navigation."""
    if not st.session_state.get('authenticated'): return

    st.sidebar.title("Navigation")
    st.sidebar.write(f"👋 Hello, {st.session_state.current_user}!")

    pages = {"Dashboard": "main", "Journal": "journal"} # Resources accessed via entry
    current_view = st.session_state.get('current_view', 'main')
    page_keys = list(pages.keys())
    try: current_index = list(pages.values()).index(current_view)
    except ValueError: current_index = 0

    selected_page_name = st.sidebar.radio("Go to:", page_keys, index=current_index, key="sidebar_nav_gem")
    selected_view = pages[selected_page_name]

    if selected_view != current_view:
        st.session_state.current_view = selected_view
        if selected_view in ["main", "journal"]: # Clear context if leaving detail view
            if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
        st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Logout", key="logout_btn_gem"):
        for key in ['authenticated', 'current_user', 'current_view', 'current_emotion_id']:
            if key in st.session_state: del st.session_state[key]
        st.session_state.current_view = "login"
        st.rerun()

def render_main_dashboard():
    """Main dashboard view."""
    st.title("🌱 Your Growth Dashboard")
    user_id = st.session_state.current_user
    emotion_history = get_user_emotion_history(user_id)

    col1, col2 = st.columns([2, 1])
    with col1: st.markdown(f"### Welcome back, {user_id}!")
    with col2:
        if st.button("📝 New Journal Entry", type="primary", use_container_width=True, key="new_entry_dash_gem"):
            st.session_state.current_view = "journal"
            if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
            st.rerun()

    st.divider()
    if not emotion_history:
        st.info("Start your journey: add your first journal entry!")
    else:
        st.subheader("Recent Emotional Journey")
        sorted_history = sorted(emotion_history, key=lambda x: x.get('timestamp', ''), reverse=True)
        for entry in sorted_history[:5]: # Show latest 5
            entry_id = entry.get('id')
            ts_str = entry.get('timestamp', '')
            display_time = ts_str[:16].replace('T', ' ') if ts_str else '?'
            analysis = entry.get('analysis', {})
            is_analysis_ok = isinstance(analysis, dict) and 'error' not in analysis and 'raw_response' not in analysis
            p_emotion = analysis.get('primary_emotion', 'Entry') if is_analysis_ok else 'Entry'
            intensity = analysis.get('intensity') if is_analysis_ok else None
            title = f"**{p_emotion}**{' (Intensity: ' + str(intensity) + ')' if intensity is not None else ''} - {display_time}"

            with st.expander(title):
                st.caption("Entry:")
                st.write(entry.get('journal_entry', 'N/A'))
                if not is_analysis_ok: st.warning("Analysis may have issues.")
                if st.button("View Analysis & Actions", key=f"view_dash_{entry_id}"):
                    st.session_state.current_emotion_id = entry_id
                    st.session_state.current_view = "emotion_analysis"
                    st.rerun()


def render_journal_page():
    """Page for creating a new journal entry."""
    st.title("📝 Emotional Journal")
    with st.form("journal_form_gem"):
        journal_entry = st.text_area("What's on your mind?", height=250, key="journal_input_gem", placeholder="Describe situation, feelings, thoughts...")
        if st.form_submit_button("Analyze My Emotions", type="primary"):
            if not journal_entry or len(journal_entry.strip()) < 10:
                st.warning("Please write a bit more for analysis.")
            else:
                with st.spinner("🧠 Analyzing with Gemini..."):
                    analysis = analyze_emotion(journal_entry.strip())
                    e_id = save_emotion_entry(st.session_state.current_user, {'journal_entry': journal_entry.strip(), 'analysis': analysis})
                    st.session_state.current_emotion_id = e_id
                    st.session_state.current_view = "emotion_analysis"
                    st.rerun()
    st.divider()
    st.subheader("Past Journal Entries")
    history = get_user_emotion_history(st.session_state.current_user)
    # Display past entries (condensed for brevity, similar to dashboard view)
    if not history: st.info("No past entries.")
    else:
        sorted_hist = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
        for entry in sorted_hist:
             e_id = entry.get('id')
             ts = entry.get('timestamp', '')[:16].replace('T', ' ')
             analysis = entry.get('analysis', {})
             is_ok = isinstance(analysis, dict) and 'error' not in analysis and 'raw_response' not in analysis
             p_emo = analysis.get('primary_emotion', 'Entry') if is_ok else 'Entry'
             prev = entry.get('journal_entry', '')[:80] + '...'
             col1, col2 = st.columns([4, 1])
             with col1: st.write(f"**{p_emo}** ({ts}): {prev}")
             with col2:
                 if st.button("Details", key=f"list_journ_{e_id}", use_container_width=True):
                     st.session_state.current_emotion_id = e_id
                     st.session_state.current_view = "emotion_analysis"
                     st.rerun()
             st.markdown("---")


def render_emotion_analysis(emotion_id: str):
    """Displays analysis and actions for an entry."""
    st.title("🧠 Emotion Analysis & Actions")
    user_id = st.session_state.current_user
    emotion_data = get_emotion_entry(user_id, emotion_id)

    if not emotion_data:
        st.error("Entry not found.")
        if st.button("Back", key="back_analysis_err"): st.session_state.current_view = "journal"; st.rerun()
        return

    st.subheader("Your Journal Entry")
    st.markdown(f"> _{emotion_data.get('journal_entry', 'N/A')}_")
    st.caption(f"Logged: {emotion_data.get('timestamp', '')[:19].replace('T', ' ')}")
    st.divider()

    st.subheader("Gemini AI Analysis")
    analysis = emotion_data.get('analysis', {})
    analysis_successful = isinstance(analysis, dict) and 'error' not in analysis and 'raw_response' not in analysis

    if not analysis: st.warning("Analysis data missing.")
    elif 'error' in analysis: st.error(f"Analysis Error: {analysis['error']}"); st.code(analysis.get('raw_response',''), language='text')
    elif 'raw_response' in analysis: st.warning("Analysis needs review:"); st.code(analysis['raw_response'], language='text')
    else: # Display successful analysis
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Primary Emotion", analysis.get('primary_emotion', 'N/A').capitalize())
            if 'intensity' in analysis: st.metric("Intensity (1-10)", analysis['intensity'])
        with col2:
            triggers = analysis.get('triggers', [])
            st.markdown("**Potential Triggers:**")
            if triggers: st.write('\n'.join(f"- {t}" for t in triggers))
            else: st.write("_None identified._")
        growth_opps = analysis.get('growth_opportunities', [])
        st.markdown("**Growth Opportunities:**")
        if growth_opps: st.write('\n'.join(f"- {g}" for g in growth_opps))
        else: st.write("_None identified._")

    st.divider()
    st.subheader("Next Steps")
    col1, col2 = st.columns(2)
    growth_plan = get_growth_plan(user_id, emotion_id)
    resources = get_resources(user_id, emotion_id)

    with col1: # Growth Plan Actions
        st.markdown("##### **Personal Growth Plan**")
        plan_exists = isinstance(growth_plan, dict) and 'error' not in growth_plan and 'raw_response' not in growth_plan
        if plan_exists:
            st.success("Growth plan generated.")
            if st.button("View Plan", key="view_plan_btn_gem", use_container_width=True):
                st.session_state.current_view = "growth_plan"; st.rerun()
        elif growth_plan: # If failed previously
            st.warning("Previous plan generation failed.")
            if st.button("💡 Regenerate Plan", key="regen_plan_gem", use_container_width=True, disabled=not analysis_successful):
                with st.spinner("✨ Regenerating plan with Gemini..."):
                    plan = generate_growth_plan(analysis, {})
                    save_growth_plan(user_id, emotion_id, plan)
                    if isinstance(plan, dict) and 'error' not in plan and 'raw_response' not in plan:
                        st.session_state.current_view = "growth_plan"
                    st.rerun()
        else: # If no plan exists
            if st.button("💡 Create Growth Plan", key="create_plan_gem", use_container_width=True, type="primary", disabled=not analysis_successful):
                with st.spinner("✨ Creating plan with Gemini..."):
                    plan = generate_growth_plan(analysis, {})
                    save_growth_plan(user_id, emotion_id, plan)
                    if isinstance(plan, dict) and 'error' not in plan and 'raw_response' not in plan:
                        st.session_state.current_view = "growth_plan"
                    st.rerun()
        if not analysis_successful: st.caption("Analysis must succeed first.")

    with col2: # Resources Actions
        st.markdown("##### **AI-Generated Resources**")
        res_exist = isinstance(resources, dict) and 'error' not in resources and 'raw_response' not in resources
        if res_exist:
            st.success("Personalized resources generated.")
            if st.button("📚 View Resources", key="view_res_btn_gem", use_container_width=True):
                st.session_state.current_view = "view_resources"; st.rerun()
        elif resources: # If failed previously
             st.warning("Previous resource generation failed.")
             if st.button("🔄 Regenerate Resources", key="regen_res_gem", use_container_width=True, disabled=not analysis_successful):
                 with st.spinner("✨ Regenerating resources with Gemini..."):
                      res = generate_resources_from_profile(analysis, growth_plan)
                      save_resource(user_id, emotion_id, res)
                      if isinstance(res, dict) and 'error' not in res and 'raw_response' not in res:
                           st.session_state.current_view = "view_resources"
                      st.rerun()
        else: # If no resources exist
            if st.button("✨ Generate Resources", key="gen_res_gem", use_container_width=True, type="primary", disabled=not analysis_successful):
                 with st.spinner("✨ Generating resources with Gemini..."):
                      res = generate_resources_from_profile(analysis, growth_plan)
                      save_resource(user_id, emotion_id, res)
                      if isinstance(res, dict) and 'error' not in res and 'raw_response' not in res:
                          st.session_state.current_view = "view_resources"
                      st.rerun()
        if not analysis_successful: st.caption("Analysis must succeed first.")


def render_growth_plan(emotion_id: str):
    """Displays the generated growth plan."""
    st.title("🚀 Your Growth Plan")
    user_id = st.session_state.current_user
    plan = get_growth_plan(user_id, emotion_id)
    emotion_data = get_emotion_entry(user_id, emotion_id)

    if not plan:
        st.error("Growth plan not found.")
        if st.button("Back", key="back_plan_err_gem"): st.session_state.current_view = "emotion_analysis"; st.rerun()
        return

    if emotion_data and isinstance(emotion_data.get('analysis'), dict):
         p_emotion = emotion_data['analysis'].get('primary_emotion', 'the situation')
         st.caption(f"Plan related to your entry about: **{p_emotion}**")
    st.divider()

    if isinstance(plan, dict) and 'error' not in plan and 'raw_response' not in plan:
        st.header(plan.get('title', 'Personal Growth Plan'))
        steps = plan.get('steps', [])
        if steps:
            st.subheader("Action Steps")
            for i, step in enumerate(steps, 1):
                 with st.expander(f"**Step {i}: {step.get('title', '')}**", expanded=True): st.write(step.get('description', ''))
        outcomes = plan.get('expected_outcomes', [])
        if outcomes:
            st.subheader("Expected Outcomes"); st.write('\n'.join(f"- {o}" for o in outcomes))
    elif 'error' in plan: st.error(f"Error in plan: {plan['error']}"); st.code(plan.get('raw_response',''), language='text')
    elif 'raw_response' in plan: st.warning("Plan needs review:"); st.code(plan['raw_response'], language='text')

    st.divider()
    # Ensure this button logic is complete
    if st.button("⬅️ Back to Analysis", key="back_plan_gem"):
        st.session_state.current_view = "emotion_analysis"
        st.rerun()

def render_view_resources(emotion_id: str):
    """Displays the AI-generated resources."""
    st.title("📚 AI-Generated Resources")
    user_id = st.session_state.current_user
    resources = get_resources(user_id, emotion_id)
    emotion_data = get_emotion_entry(user_id, emotion_id)

    if not resources:
        st.error("Resources not found for this entry.")
        if st.button("Back", key="back_res_err_gem"): st.session_state.current_view = "emotion_analysis"; st.rerun()
        return

    if emotion_data and isinstance(emotion_data.get('analysis'), dict):
        p_emotion = emotion_data['analysis'].get('primary_emotion', 'this situation')
        st.caption(f"Resources generated based on your entry about: **{p_emotion}**")
    st.divider()

    if isinstance(resources, dict) and 'error' not in resources and 'raw_response' not in resources:
        insights = resources.get('key_insights', [])
        if insights: st.subheader("💡 Key Insights"); st.write('\n'.join(f"- {i}" for i in insights))

        exercises = resources.get('practical_exercises', [])
        if exercises: st.subheader("🧘 Practical Exercises"); st.write('\n'.join(f"{n}. {e}" for n, e in enumerate(exercises, 1)))

        readings = resources.get('recommended_readings', [])
        if readings:
            st.subheader("📖 Recommended Reading")
            for r in readings: st.markdown(f"**{r.get('title', '')}:** {r.get('description', '')}")
    elif 'error' in resources: st.error(f"Error in resources: {resources['error']}"); st.code(resources.get('raw_response',''), language='text')
    elif 'raw_response' in resources: st.warning("Resources need review:"); st.code(resources['raw_response'], language='text')

    st.divider()
    if st.button("⬅️ Back to Analysis", key="back_res_gem"):
        st.session_state.current_view = "emotion_analysis"
        st.rerun()


# --- Main Application Logic ---
def main():
    """Runs the Streamlit application."""
    st.set_page_config(page_title="Pathly MVP (Gemini)", layout="wide", initial_sidebar_state="auto")

    # Initialize state (checks for API key internally now)
    init_state()

    # Routing
    if not st.session_state.authenticated:
        st.session_state.current_view = "login" # Force login view if not auth
        render_login_page()
    else:
        render_sidebar() # Show nav for logged-in users
        view = st.session_state.current_view
        emotion_id = st.session_state.get('current_emotion_id')

        # Main content rendering based on view
        if view == "main": render_main_dashboard()
        elif view == "journal": render_journal_page()
        elif view == "emotion_analysis":
            if emotion_id: render_emotion_analysis(emotion_id)
            else: st.warning("No entry selected."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "growth_plan":
            if emotion_id: render_growth_plan(emotion_id)
            else: st.warning("No entry selected."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "view_resources":
            if emotion_id: render_view_resources(emotion_id)
            else: st.warning("No entry selected."); st.session_state.current_view = "journal"; st.rerun()
        else: # Fallback
            st.error(f"Unknown view state: {view}. Returning to dashboard.")
            st.session_state.current_view = "main"
            if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
            st.rerun()

if __name__ == "__main__":
    main()
