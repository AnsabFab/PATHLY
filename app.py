# pathly_mvp_gemini_v2.py
import os
import json
import datetime
import uuid
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import pytz # Required by google-generativeai sometimes, good practice

# --- PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Aishura MVP (Gemini)", # Renamed to match spec
    layout="wide",
    initial_sidebar_state="auto"
)
# --- END PAGE CONFIG ---

# --- Configuration & API Key Check ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Perform checks AFTER page config
if not GEMINI_API_KEY:
    st.error("🔴 Error: GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Note: Using gemini-1.5-flash-latest model
    MODEL_NAME = "gemini-2.0-flash"
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(MODEL_NAME)
    # Only show success discreetly if needed, avoid cluttering main screen
    # st.sidebar.success(f"Gemini API OK")
except Exception as e:
    st.error(f"🔴 Error configuring Gemini API: {e}")
    st.stop()
# --- End Configuration ---


# --- LLM Prompts (Revised for Aishura Spec) ---

EMOTION_PROMPT = """SYSTEM: You are Aishura, an empathetic AI career assistant. Analyze the user's entry for their primary career-related emotion, intensity (1-10), potential triggers, and growth opportunities. Respond ONLY with a valid JSON object. No explanations.
USER: Analyze this entry: {journal_entry_text}
JSON Structure: {"primary_emotion": "string", "intensity": integer, "triggers": ["string"], "growth_opportunities": ["string"]}
"""

GROWTH_PLAN_PROMPT = """SYSTEM: You are Aishura, an AI career coach. Create a concise, actionable growth plan based on the user's career concern analysis. Respond ONLY with a valid JSON object. No explanations.
USER: Create a plan based on this analysis: {analysis_and_goals_json}
JSON Structure: {"title": "string", "steps": [{"title": "string", "description": "string"}], "expected_outcomes": ["string"]}
"""

# Revised Prompt for Actionable Suggestions (Replaces Resources/Local Jobs)
ACTION_SUGGESTION_PROMPT = """SYSTEM: You are Aishura, an AI career assistant providing proactive, actionable advice. Respond ONLY with a valid JSON object. No explanations.
USER: Based on the user's career concern analysis and growth plan, suggest concrete next steps. Provide:
1. "immediate_actions": [List of 2-3 specific, small actions the user can take now related to their career goal (e.g., 'Draft 3 bullet points for your resume about project X', 'Spend 15 mins researching companies in [Industry]', 'Identify one person to reach out to for networking')].
2. "preparation_guidance": [{"item": "e.g., Resume Update", "guidance": "Focus on quantifying achievements using the STAR method..."}]. Include guidance for 1-2 key preparation items relevant to their situation.
3. "key_insight": "A single, concise, encouraging insight related to their situation."
JSON Structure: {"immediate_actions": ["string"], "preparation_guidance": [{"item": "string", "guidance": "string"}], "key_insight": "string"}

User Profile (Analysis & Plan):
{profile_data_json}
"""

# --- Session State Management ---
def init_state():
    if 'user_db' not in st.session_state: st.session_state.user_db = {}
    # Using 'entry_db' instead of 'emotion_db' for clarity
    if 'entry_db' not in st.session_state: st.session_state.entry_db = {}
    if 'plans_db' not in st.session_state: st.session_state.plans_db = {}
    # Using 'actions_db' instead of 'resource_db'
    if 'actions_db' not in st.session_state: st.session_state.actions_db = {}

    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'current_user' not in st.session_state: st.session_state.current_user = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "login" if not st.session_state.authenticated else "main"
    # Using 'current_entry_id' for clarity
    if 'current_entry_id' not in st.session_state: st.session_state.current_entry_id = None

# --- Data Persistence Functions ---
def save_user_data(user_id: str, data: Dict): st.session_state.user_db[user_id] = data

def save_journal_entry(user_id: str, entry_data: Dict):
    if user_id not in st.session_state.entry_db: st.session_state.entry_db[user_id] = []
    entry_data['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry_data['id'] = str(uuid.uuid4())
    st.session_state.entry_db[user_id].append(entry_data)
    return entry_data['id']

def get_user_entry_history(user_id: str) -> List[Dict]: return st.session_state.entry_db.get(user_id, [])

def get_journal_entry(user_id: str, entry_id: str) -> Optional[Dict]:
    for entry in get_user_entry_history(user_id):
        if entry.get('id') == entry_id: return entry
    return None

def save_growth_plan(user_id: str, entry_id: str, plan_data: Dict):
    if user_id not in st.session_state.plans_db: st.session_state.plans_db[user_id] = {}
    st.session_state.plans_db[user_id][entry_id] = plan_data

def get_growth_plan(user_id: str, entry_id: str) -> Optional[Dict]:
    return st.session_state.plans_db.get(user_id, {}).get(entry_id)

# Renamed functions for action suggestions
def save_action_suggestions(user_id: str, entry_id: str, action_data: Dict):
    if user_id not in st.session_state.actions_db: st.session_state.actions_db[user_id] = {}
    st.session_state.actions_db[user_id][entry_id] = action_data

def get_action_suggestions(user_id: str, entry_id: str) -> Optional[Dict]:
    return st.session_state.actions_db.get(user_id, {}).get(entry_id)

# --- LLM Interaction Logic (Using Gemini API) ---
def call_gemini_llm(prompt: str) -> Dict:
    response_data = {"raw_response": None, "error": None}
    try:
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        # Check for safety blocks before accessing text
        if not response.candidates:
             response_data["error"] = "Content blocked by safety filters or other reason."
             st.error(response_data["error"] + f" Finish reason: {response.prompt_feedback.block_reason}")
             return response_data

        raw_text = response.text
        response_data["raw_response"] = raw_text

        try:
            parsed_json = json.loads(raw_text)
            return parsed_json
        except json.JSONDecodeError:
            cleaned_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                parsed_json = json.loads(cleaned_text)
                return parsed_json
            except json.JSONDecodeError as json_err_clean:
                 response_data["error"] = f"Failed to parse JSON even after cleaning: {json_err_clean}"
                 st.warning(f"{response_data['error']} Raw: {raw_text}")
        except Exception as parse_err:
             response_data["error"] = f"Unexpected error processing LLM response content: {parse_err}"
             st.error(f"{response_data['error']} Raw: {raw_text}")

    except Exception as e:
        response_data["error"] = f"Gemini API call failed: {e}"
        st.error(f"{response_data['error']}")

    return response_data

def analyze_entry(journal_entry: str) -> Dict:
    prompt = EMOTION_PROMPT.format(journal_entry_text=journal_entry)
    return call_gemini_llm(prompt)

def generate_growth_plan(emotion_analysis: Dict, user_goals: Optional[Dict] = None) -> Dict:
    input_data = {
        "emotion_analysis": emotion_analysis,
        "user_goals": user_goals or {"general_goal": "Improve career situation and well-being"}
    }
    input_json_str = json.dumps(input_data, indent=2)
    prompt = GROWTH_PLAN_PROMPT.format(analysis_and_goals_json=input_json_str)
    return call_gemini_llm(prompt)

# Renamed function for career actions
def generate_career_actions(emotion_analysis: Dict, growth_plan: Optional[Dict]) -> Dict:
    if not emotion_analysis: return {"error": "Cannot generate actions without analysis."}
    profile_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan or {"message": "No growth plan available."},
    }
    input_json_str = json.dumps(profile_data, indent=2)
    prompt = ACTION_SUGGESTION_PROMPT.format(profile_data_json=input_json_str)
    return call_gemini_llm(prompt)

# --- UI Rendering Functions ---

def render_login_page():
    st.title("✨ Aishura MVP")
    st.subheader("Your Emotionally Intelligent AI Career Assistant")
    st.markdown("*(Investor Demo - Gemini Powered)*")
    col1, col2 = st.columns(2)
    with col1: # Login
        st.subheader("Login")
        with st.form("login_form_aishura"):
            uname = st.text_input("Username", key="login_u_aishura")
            pwd = st.text_input("Password", type="password", key="login_p_aishura")
            if st.form_submit_button("Login"):
                # Basic auth check
                if uname in st.session_state.user_db and st.session_state.user_db[uname].get('password') == pwd:
                    st.session_state.current_user = uname; st.session_state.authenticated = True; st.session_state.current_view = "main"; st.rerun()
                else: st.error("Invalid credentials.")
    with col2: # Signup
        st.subheader("Sign Up")
        with st.form("signup_form_aishura"):
            s_uname = st.text_input("Choose Username", key="signup_u_aishura")
            s_pwd = st.text_input("Choose Password", type="password", key="signup_p_aishura")
            if st.form_submit_button("Sign Up"):
                if not s_uname or not s_pwd: st.warning("Username/password required.")
                elif s_uname in st.session_state.user_db: st.error("Username taken.")
                elif len(s_pwd) < 4: st.error("Password too short (min 4 chars).")
                else:
                    st.session_state.user_db[s_uname] = {'password': s_pwd} # Store plain pass for demo ONLY
                    st.success("Account created! Please login.")

def render_sidebar():
    if not st.session_state.get('authenticated'): return
    st.sidebar.title("Aishura Menu")
    st.sidebar.write(f"👤 Welcome, {st.session_state.current_user}!")
    pages = {"Dashboard": "main", "Journal": "journal"}
    current_view = st.session_state.get('current_view', 'main')
    page_keys = list(pages.keys())
    try: current_index = list(pages.values()).index(current_view)
    except ValueError: current_index = 0
    selected_page_name = st.sidebar.radio("Navigation", page_keys, index=current_index, key="sidebar_nav_aishura")
    selected_view = pages[selected_page_name]
    if selected_view != current_view:
        st.session_state.current_view = selected_view
        if selected_view in ["main", "journal"]: # Clear context
            if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
        st.rerun()
    st.sidebar.divider()
    if st.sidebar.button("Logout", key="logout_aishura"):
        for key in ['authenticated', 'current_user', 'current_view', 'current_entry_id']:
            if key in st.session_state: del st.session_state[key]
        st.session_state.current_view = "login"; st.rerun()

def render_main_dashboard():
    st.title("✨ Aishura Dashboard")
    user_id = st.session_state.current_user
    entry_history = get_user_entry_history(user_id)
    col1, col2 = st.columns([2, 1])
    with col1: st.markdown(f"### How can I help you today, {user_id}?")
    with col2:
        if st.button("📝 Share Your Thoughts/Concerns", type="primary", use_container_width=True, key="new_entry_dash_aishura"):
            st.session_state.current_view = "journal";
            if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
            st.rerun()
    st.divider()
    if not entry_history: st.info("Click the button above to share how you're feeling or what career challenge you're facing.")
    else:
        st.subheader("Recent Entries & Insights")
        sorted_history = sorted(entry_history, key=lambda x: x.get('timestamp', ''), reverse=True)
        for entry in sorted_history[:3]: # Show latest 3
            entry_id = entry.get('id'); ts_str = entry.get('timestamp', '')
            display_time = ts_str[:16].replace('T', ' ') if ts_str else '?'
            analysis = entry.get('analysis', {}); is_analysis_ok = isinstance(analysis, dict) and 'error' not in analysis
            p_emotion = analysis.get('primary_emotion', 'Entry') if is_analysis_ok else 'Entry'
            title = f"**{p_emotion}** - {display_time}"
            with st.expander(title):
                st.caption("Your Entry:"); st.write(entry.get('journal_entry', 'N/A'))
                if not is_analysis_ok: st.warning("Analysis has issues.")
                if st.button("View Details & Actions", key=f"view_dash_aishura_{entry_id}"):
                    st.session_state.current_entry_id = entry_id; st.session_state.current_view = "analysis"; st.rerun()

def render_journal_page():
    st.title("📝 How are you feeling about your career right now?")
    st.write("Share your current state, concerns, or goals. (e.g., 'Feeling unmotivated', 'Job search isn't working', 'Want to get into a big tech company')")
    with st.form("journal_form_aishura"):
        journal_entry = st.text_area("Share here:", height=150, key="journal_input_aishura", placeholder="What's on your mind?")
        if st.form_submit_button("Send to Aishura", type="primary"):
            if not journal_entry or len(journal_entry.strip()) < 5: st.warning("Please share a bit more.")
            else:
                with st.spinner("🧠 Understanding your situation..."):
                    analysis = analyze_entry(journal_entry.strip())
                    e_id = save_journal_entry(st.session_state.current_user, {'journal_entry': journal_entry.strip(), 'analysis': analysis})
                    st.session_state.current_entry_id = e_id; st.session_state.current_view = "analysis"; st.rerun()
    st.divider(); st.subheader("Past Entries")
    history = get_user_entry_history(st.session_state.current_user)
    if not history: st.info("No past entries.")
    else: # Condensed past entries display
        for entry in sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]:
             e_id = entry.get('id'); ts = entry.get('timestamp', '')[:16].replace('T', ' '); analysis = entry.get('analysis', {})
             is_ok = isinstance(analysis, dict) and 'error' not in analysis
             p_emo = analysis.get('primary_emotion', 'Entry') if is_ok else 'Entry'; prev = entry.get('journal_entry', '')[:70] + '...'
             col1, col2 = st.columns([4, 1])
             with col1: st.write(f"*{p_emo}* ({ts}): {prev}")
             with col2:
                 if st.button("Details", key=f"list_journ_aishura_{e_id}", use_container_width=True):
                     st.session_state.current_entry_id = e_id; st.session_state.current_view = "analysis"; st.rerun()
             st.markdown("---")

# Renamed from render_emotion_analysis
def render_analysis_page(entry_id: str):
    st.title("💬 Aishura's Understanding & Next Steps")
    user_id = st.session_state.current_user; entry_data = get_journal_entry(user_id, entry_id)
    if not entry_data:
        st.error("Entry not found.");
        if st.button("Back", key="back_analysis_err_aishura"): st.session_state.current_view = "journal"; st.rerun()
        return

    st.subheader("Your Input")
    st.markdown(f"> _{entry_data.get('journal_entry', 'N/A')}_")
    st.caption(f"Received: {entry_data.get('timestamp', '')[:19].replace('T', ' ')}")
    st.divider()

    analysis = entry_data.get('analysis', {})
    analysis_successful = isinstance(analysis, dict) and 'error' not in analysis and 'raw_response' not in analysis

    # Empathetic Response (Based on Notion Spec)
    st.subheader("Aishura's Response")
    if analysis_successful and analysis.get('primary_emotion'):
        empathetic_response = f"I understand you're feeling **{analysis.get('primary_emotion', 'this way')}**. Many people experience this, and it's okay. Let's see how we can move forward together."
        st.success(empathetic_response) # Using success box for emphasis
    else:
        st.info("Let's break down your situation and find a path forward.") # Generic fallback

    # Display Analysis Details
    st.markdown("**Analysis Details:**")
    if not analysis: st.warning("Analysis data missing.")
    elif 'error' in analysis: st.error(f"Analysis Error: {analysis['error']}"); st.code(analysis.get('raw_response',''), language='text')
    elif 'raw_response' in analysis: st.warning("Analysis needs review:"); st.code(analysis['raw_response'], language='text')
    else: # Display successful analysis
        # Using columns for better layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Primary Emotion", analysis.get('primary_emotion', 'N/A').capitalize())
            if 'intensity' in analysis: st.metric("Intensity (1-10)", analysis['intensity'])
        with col2:
            triggers = analysis.get('triggers', [])
            st.markdown("**Potential Triggers:**")
            if triggers: st.write('\n'.join(f"- {t}" for t in triggers)) # Fixed syntax here
            else: st.write("_None identified._")

        growth_opps = analysis.get('growth_opportunities', [])
        st.markdown("**Growth Opportunities:**")
        if growth_opps: st.write('\n'.join(f"- {g}" for g in growth_opps)) # Fixed syntax here
        else: st.write("_None identified._")

    st.divider()
    st.subheader("Suggested Next Steps")
    col1, col2 = st.columns(2)
    growth_plan = get_growth_plan(user_id, entry_id)
    action_suggestions = get_action_suggestions(user_id, entry_id) # Use new name

    with col1: # Growth Plan Actions
        st.markdown("##### **1. Personal Growth Plan**")
        plan_exists = isinstance(growth_plan, dict) and 'error' not in growth_plan
        if plan_exists: st.success("Personalized plan generated.")
        elif growth_plan: st.warning("Previous plan generation failed.")

        # Combine view/generate buttons
        plan_btn_text = "View Plan" if plan_exists else ("Regenerate Plan" if growth_plan else "Create Growth Plan")
        plan_btn_key = "view_regen_create_plan_aishura"
        plan_btn_type = "secondary" if growth_plan else "primary"
        if st.button(f"🚀 {plan_btn_text}", key=plan_btn_key, use_container_width=True, type=plan_btn_type, disabled=not analysis_successful):
            if plan_exists: # Just navigate if exists
                 st.session_state.current_view = "growth_plan"; st.rerun()
            else: # Generate/Regenerate
                with st.spinner("✨ Crafting your plan..."):
                    plan = generate_growth_plan(analysis, {})
                    save_growth_plan(user_id, entry_id, plan)
                if isinstance(plan, dict) and 'error' not in plan: st.session_state.current_view = "growth_plan"
                st.rerun() # Rerun to show result or navigate

        if not analysis_successful: st.caption("Analysis must succeed first.")

    with col2: # Action Suggestion Actions
        st.markdown("##### **2. Actionable Suggestions**")
        actions_exist = isinstance(action_suggestions, dict) and 'error' not in action_suggestions
        if actions_exist: st.success("Action suggestions ready.")
        elif action_suggestions: st.warning("Previous suggestions failed.")

        action_btn_text = "View Actions" if actions_exist else ("Regenerate Actions" if action_suggestions else "Get Action Suggestions")
        action_btn_key = "view_regen_create_action_aishura"
        action_btn_type = "secondary" if action_suggestions else "primary"
        if st.button(f"💡 {action_btn_text}", key=action_btn_key, use_container_width=True, type=action_btn_type, disabled=not analysis_successful):
            if actions_exist: # Just navigate
                st.session_state.current_view = "view_actions"; st.rerun()
            else: # Generate/Regenerate
                with st.spinner("✨ Generating actionable advice..."):
                    actions = generate_career_actions(analysis, growth_plan) # Pass analysis & plan
                    save_action_suggestions(user_id, entry_id, actions)
                if isinstance(actions, dict) and 'error' not in actions: st.session_state.current_view = "view_actions"
                st.rerun() # Rerun to show result or navigate

        if not analysis_successful: st.caption("Analysis must succeed first.")

# This function remains largely the same, displaying the plan
def render_growth_plan(entry_id: str):
    st.title("🚀 Your Personal Growth Plan")
    user_id = st.session_state.current_user; plan = get_growth_plan(user_id, entry_id); entry_data = get_journal_entry(user_id, entry_id)
    if not plan: st.error("Plan not found.");
    if st.button("Back", key="back_plan_err_aishura"): st.session_state.current_view = "analysis"; st.rerun(); return
    if entry_data and isinstance(entry_data.get('analysis'), dict): p_emotion = entry_data['analysis'].get('primary_emotion', 'situation'); st.caption(f"Plan related to: **{p_emotion}**")
    st.divider()
    if isinstance(plan, dict) and 'error' not in plan:
        st.header(plan.get('title', 'Growth Plan'))
        steps = plan.get('steps', [])
        if steps: st.subheader("Action Steps");
        for i, step in enumerate(steps, 1):
            with st.expander(f"**Step {i}: {step.get('title', '')}**", expanded=True): st.write(step.get('description', ''))
        outcomes = plan.get('expected_outcomes', [])
        if outcomes: st.subheader("Expected Outcomes"); st.write('\n'.join(f"- {o}" for o in outcomes))
    elif 'error' in plan: st.error(f"Error in plan: {plan['error']}"); st.code(plan.get('raw_response',''), language='text')
    st.divider()
    if st.button("⬅️ Back to Analysis & Actions", key="back_plan_aishura"): st.session_state.current_view = "analysis"; st.rerun()

# Renamed from render_view_resources
def render_view_actions(entry_id: str):
    st.title("💡 Aishura's Action Suggestions")
    st.subheader("Here are some concrete steps based on our conversation:")
    user_id = st.session_state.current_user; actions = get_action_suggestions(user_id, entry_id); entry_data = get_journal_entry(user_id, entry_id)
    if not actions: st.error("Action suggestions not found.");
    if st.button("Back", key="back_action_err_aishura"): st.session_state.current_view = "analysis"; st.rerun(); return

    if entry_data and isinstance(entry_data.get('analysis'), dict): p_emotion = entry_data['analysis'].get('primary_emotion', 'your situation'); st.caption(f"Suggestions related to: **{p_emotion}**")
    st.divider()

    # Display the structured action suggestions
    if isinstance(actions, dict) and 'error' not in actions:
        insight = actions.get('key_insight')
        if insight:
            st.markdown("#### Key Insight ✨")
            st.info(insight) # Use info box for insight

        immediate = actions.get('immediate_actions', [])
        if immediate:
            st.markdown("#### Immediate Next Steps ⚡")
            st.write('\n'.join(f"- {a}" for a in immediate))

        prep_guidance = actions.get('preparation_guidance', [])
        if prep_guidance:
            st.markdown("#### Preparation Guidance 📝")
            for item in prep_guidance:
                st.markdown(f"**{item.get('item', 'Item')}:** {item.get('guidance', '')}")

        # Mimic Action Card Summary Idea
        st.divider()
        st.markdown("##### **Action Summary Card (Example)**")
        st.markdown("- **Focus:** Addressing your concern about " + f"**{p_emotion}**" if p_emotion else "your situation")
        st.markdown("- **Next Steps:** " + ", ".join(immediate[:2]) + ("..." if len(immediate)>2 else ""))
        prep_items = ", ".join([item.get('item', '') for item in prep_guidance])
        if prep_items: st.markdown(f"- **Prep Focus:** {prep_items}")


    elif 'error' in actions: st.error(f"Error generating suggestions: {actions['error']}"); st.code(actions.get('raw_response',''), language='text')

    st.divider()
    if st.button("⬅️ Back to Analysis & Actions", key="back_action_aishura"): st.session_state.current_view = "analysis"; st.rerun()


# --- Main Application Logic ---
def main():
    init_state() # Initialize session state variables
    if not st.session_state.authenticated:
        st.session_state.current_view = "login" # Force login view if not auth
        render_login_page()
    else:
        render_sidebar() # Show nav for logged-in users
        view = st.session_state.current_view
        entry_id = st.session_state.get('current_entry_id') # Use updated key name

        # Main content rendering based on view
        if view == "main": render_main_dashboard()
        elif view == "journal": render_journal_page()
        elif view == "analysis": # Renamed view
            if entry_id: render_analysis_page(entry_id) # Use renamed function
            else: st.warning("No entry selected."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "growth_plan":
            if entry_id: render_growth_plan(entry_id)
            else: st.warning("No entry selected."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "view_actions": # Renamed view
            if entry_id: render_view_actions(entry_id) # Use renamed function
            else: st.warning("No entry selected."); st.session_state.current_view = "journal"; st.rerun()
        else: # Fallback
            st.error(f"Unknown view state: {view}. Returning to dashboard.")
            st.session_state.current_view = "main";
            if 'current_entry_id' in st.session_state: del st.session_state['current_entry_id']
            st.rerun()

if __name__ == "__main__":
    main()
