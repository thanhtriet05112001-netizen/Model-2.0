import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import re
import csv
import os
import json
import time
from datetime import datetime, timezone, timedelta

# Import NLTK & download essential light tokenizers safely inside Streamlit
import nltk

@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)

setup_nltk()
from nltk.tokenize import sent_tokenize, word_tokenize

# ==========================================
# 1. SETUP, MODERN STYLING & SKELETON ANIMATIONS
# ==========================================
st.set_page_config(page_title="C.O.W", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Highlight Styles */
    .error-highlight { background-color: #ffe6e6; color: #b30000; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ff9999; cursor: help; text-decoration: line-through; }
    .correction-preview { color: #cc0000; font-size: 0.9em; margin-left: 4px; font-weight: bold; }
    .fixed-highlight { background-color: #e6ffe6; color: #006600; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #99cc99; cursor: help; }

    /* Custom Loading Modal Overlay */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 10px rgba(0,102,204,0.2); }
        50% { box-shadow: 0 0 25px rgba(0,102,204,0.6); }
        100% { box-shadow: 0 0 10px rgba(0,102,204,0.2); }
    }
    
    .loading-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        border: 1px solid #e0e0e0;
        animation: pulse-glow 2s infinite ease-in-out;
        margin: 20px 0;
    }

    .loading-bar-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 15px;
    }

    .loading-bar-progress {
        width: 100%;
        height: 6px;
        background: linear-gradient(90deg, #4a90e2, #50e3c2);
        animation: shimmer 1.5s infinite linear;
        background-size: 200% 100%;
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
</style>
""", unsafe_allow_html=True)

vn_tz = timezone(timedelta(hours=7))
DEFAULT_MODEL = "openrouter/free"
LOG_FILE = "interaction_logs.json"

# Initialize Session State
if "telemetry_events" not in st.session_state:
    st.session_state.telemetry_events = []
if "last_action_timestamp" not in st.session_state:
    st.session_state.last_action_timestamp = time.time()

# ==========================================
# 2. AUTOMATIC TELEMETRY & DATA LOGGING
# ==========================================
def log_telemetry_event(event_type, details=""):
    """Tracks every user interaction and calculates exact pause/inactivity duration since last action."""
    now = time.time()
    pause_duration = round(now - st.session_state.last_action_timestamp, 2)
    st.session_state.last_action_timestamp = now
    
    event_entry = {
        "timestamp": datetime.now(vn_tz).strftime("%H:%M:%S.%f")[:-3],
        "event_type": event_type,
        "pause_before_action_seconds": pause_duration,
        "details": details
    }
    st.session_state.telemetry_events.append(event_entry)
    auto_save_session_log()

def auto_save_session_log():
    """Automatically persists interaction metrics and event logs without requiring user clicks."""
    if not st.session_state.get("evaluated", False):
        return

    eval_data = st.session_state.eval_data
    metrics = st.session_state.get("metrics", {})
    edits_list = eval_data.get("edits", [])
    
    accepted_edits = [
        f"{edit['original']}->{edit['correction']}"
        for i, edit in enumerate(edits_list)
        if st.session_state.get(f"edit_{i}", False)
    ]

    log_entry = {
        "timestamp_vn": datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S"),
        "task_name": st.session_state.get("task_name", "N/A"),
        "word_count": metrics.get("words", 0),
        "sentence_count": metrics.get("sents", 0),
        "wpm": metrics.get("wpm", 0),
        "lexical_diversity": metrics.get("ttr", 0),
        "accepted_edits_count": len(accepted_edits),
        "total_edits_count": len(edits_list),
        "accepted_edit_details": " | ".join(accepted_edits) if accepted_edits else "None",
        "raw_interaction_telemetry": st.session_state.telemetry_events,
        "full_draft": st.session_state.get("original_text", "")
    }

    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    # Update current session log if it exists, otherwise append
    session_id = st.session_state.get("session_id", str(time.time()))
    st.session_state.session_id = session_id
    
    updated = False
    for i, entry in enumerate(logs):
        if entry.get("session_id") == session_id:
            entry.update(log_entry)
            updated = True
            break
            
    if not updated:
        log_entry["session_id"] = session_id
        logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

def calculate_nltk_metrics(text, start_time):
    tokens = word_tokenize(text)
    words = [w.lower() for w in tokens if w.isalnum()]
    word_count = len(words)
    
    sentences = sent_tokenize(text)
    sentence_count = len(sentences) if sentences else 1
    
    elapsed_time = max(time.time() - start_time, 1)
    wpm = (word_count / elapsed_time) * 60
    
    unique_words = set(words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    return word_count, sentence_count, wpm, ttr

def safe_html_replace(text, original, html_replacement):
    escaped_orig = re.escape(original)
    pattern = r'(?<![a-zA-Z])' + escaped_orig + r'(?![a-zA-Z])'
    return re.sub(pattern, html_replacement, text, count=1)

# ==========================================
# 3. OPENROUTER API HANDLER
# ==========================================
def call_openrouter_api(messages, user_api_key=None):
    api_key = None
    if user_api_key and user_api_key.strip():
        api_key = user_api_key.strip()
    elif st.session_state.get("is_admin_authenticated", False):
        api_key = os.environ.get("OPENROUTER_API_KEY") or (
            st.secrets.get("OPENROUTER_API_KEY") if os.path.exists(".streamlit/secrets.toml") else None
        )
    
    if not api_key:
        return None, "🔑 **API Key Required**: Please enter your OpenRouter API key in the sidebar."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cow-writing-app.render.com",
        "X-Title": "Companion in Writing"
    }

    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )
        res_data = response.json()
        
        if "choices" in res_data and len(res_data["choices"]) > 0:
            return res_data["choices"][0]["message"], None
        else:
            error_msg = res_data.get("error", {}).get("message", "Unknown OpenRouter error")
            return None, f"API Error: {error_msg}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def get_ai_evaluation(user_key, task_name, task_prompt, student_text, image_base64=None):
    rubric = "Task Achievement, Coherence & Cohesion, Lexical Resource, Grammatical Range." if "1" in task_name else "Task Response, Coherence & Cohesion, Lexical Resource, Grammatical Range."

    prompt_content = f"""
    You are 'Companion in Writing', an expert IELTS examiner and proactive writing coach.
    Task: {task_name}
    Prompt: {task_prompt}
    Rubric: {rubric}
    Student Text: {student_text}

    IMPORTANT: Return ONLY a valid JSON object matching this structure:
    {{
        "band_score": "Estimated IELTS Band (e.g., 6.5)",
        "overall_feedback": "A short, encouraging paragraph summarizing strengths and weaknesses.",
        "edits": [
            {{
                "original": "Include 2 to 4 words from text for context. MUST match original text exactly.",
                "correction": "The corrected phrase",
                "explanation": "Brief explanation of the rule"
            }}
        ],
        "coach_opening_chat": "A friendly question asking the student about a specific error."
    }}
    Do NOT include markdown formatting outside the JSON block.
    """
    
    user_payload = [{"type": "text", "text": prompt_content}]
    if image_base64:
        user_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    message_data, error = call_openrouter_api([{"role": "user", "content": user_payload}], user_api_key=user_key)
    if error:
        st.error(error)
        return None

    raw_content = message_data.get("content", "{}")
    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    cleaned_content = json_match.group(0) if json_match else raw_content

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        st.error("Model failed to return valid JSON format. Please try submitting again.")
        return None

# ==========================================
# 4. HEADER & SIDEBAR NAVIGATION
# ==========================================
st.title("✨ C.O.W — Companion in Writing")
st.markdown("Your interactive IELTS workspace. Draft, review, and collaborate with your AI coach.")

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

with st.sidebar:
    st.subheader("🔑 OpenRouter API Key")
    user_api_key = st.text_input("Enter Your Key", type="password", help="Get a free key at https://openrouter.ai/keys")
    st.session_state.user_api_key = user_api_key
    
    if st.session_state.get("is_admin_authenticated", False):
        st.success("🔓 Admin Mode Active")
    elif not user_api_key:
        st.warning("⚠️ Key required for evaluation")

with st.popover("⚙️ Admin Tools"):
    admin_pass = st.text_input("Admin Password", type="password")
    expected_admin_pass = os.environ.get("ADMIN_PASSWORD") or (
        st.secrets.get("ADMIN_PASSWORD") if os.path.exists(".streamlit/secrets.toml") else "secret123"
    )
    if admin_pass == expected_admin_pass:
        st.session_state.is_admin_authenticated = True
        st.success("Authenticated as Admin.")
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button("📥 Download JSON Telemetry Logs", f, file_name=f"writing_logs_{datetime.now(vn_tz).strftime('%Y%m%d')}.json", mime="application/json")
    else:
        st.session_state.is_admin_authenticated = False

# ==========================================
# 5. DRAFTING & SKELETON LOADING UI
# ==========================================
with st.expander("📝 1. Task Setup & Drafting", expanded=not st.session_state.get("evaluated", False)):
    col_a, col_b = st.columns([1, 2])
    with col_a:
        task_name = st.selectbox("IELTS Task", ["IELTS Task 1 (Academic)", "IELTS Task 2 (Essay)"])
        uploaded_image = st.file_uploader("Upload Chart (Task 1)", type=["jpg", "png"])
        task_prompt = st.text_area("Prompt Context", placeholder="Enter the prompt here...", height=100)
    with col_b:
        student_text = st.text_area("Your Draft", placeholder="Start typing your essay here...", height=250)
    
    if st.button("Submit for Evaluation", type="primary"):
        if student_text.strip():
            log_telemetry_event("SUBMIT_DRAFT", f"Task: {task_name}, WordCount: {len(student_text.split())}")
            
            # POLISHED LOADING SKELETON OVERLAY
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown("""
                <div class="loading-card">
                    <h3 style="color: #0066cc; margin-bottom: 8px;">🎓 Analyzing Writing Performance</h3>
                    <p style="color: #666; font-size: 0.95em;">Applying IELTS criteria & evaluating grammatical structures...</p>
                    <div class="loading-bar-container">
                        <div class="loading-bar-progress"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            image_b64 = base64.b64encode(uploaded_image.read()).decode("utf-8") if uploaded_image else None
            eval_data = get_ai_evaluation(st.session_state.get("user_api_key"), task_name, task_prompt, student_text, image_b64)
            
            loading_placeholder.empty()

            if eval_data:
                st.session_state.eval_data = eval_data
                st.session_state.original_text = student_text
                st.session_state.task_name = task_name
                st.session_state.evaluated = True
                st.session_state.messages = [{"role": "assistant", "content": eval_data["coach_opening_chat"]}]
                
                for i in range(len(eval_data.get("edits", []))):
                    st.session_state[f"edit_{i}"] = False
                
                words, sents, wpm, ttr = calculate_nltk_metrics(student_text, st.session_state.start_time)
                st.session_state.metrics = {"words": words, "sents": sents, "wpm": wpm, "ttr": ttr}
                auto_save_session_log()
                st.rerun()

# ==========================================
# 6. FRAGMENTED INTERACTIVE WORKSPACE (NO FULL RELOAD)
# ==========================================
if st.session_state.get("evaluated", False):
    eval_data = st.session_state.eval_data
    
    st.markdown("---")
    st.subheader(f"🏆 Estimated Score: **{eval_data.get('band_score', 'N/A')}**")
    st.info(eval_data.get('overall_feedback', ''))
    
    work_col, chat_col = st.columns([1.5, 1])

    # Fragmented workspace ensures toggling edits DOES NOT reload the entire web app
    @st.fragment
    def render_revision_workspace():
        st.markdown("### 🔍 Interactive Revisions")
        st.caption("Check a box to accept a correction. Hover over highlighted text in the draft below for explanations.")
        
        edits_list = eval_data.get("edits", [])
        sorted_edits = sorted(edits_list, key=lambda x: len(x['original']), reverse=True)
        
        for i, edit in enumerate(sorted_edits):
            cb_key = f"edit_{i}"
            prev_val = st.session_state.get(cb_key, False)
            is_accepted = st.checkbox(f"**Fix:** {edit['original']} ➔ {edit['correction']}", key=cb_key)
            
            # Automatically record state toggles and pause intervals
            if is_accepted != prev_val:
                action_name = "ACCEPT_CORRECTION" if is_accepted else "REJECT_CORRECTION"
                log_telemetry_event(action_name, f"Edit: {edit['original']} -> {edit['correction']}")
                
            st.caption(f"*Why?* {edit['explanation']}")
        
        display_text = st.session_state.original_text
        for i, edit in enumerate(sorted_edits):
            safe_explanation = edit['explanation'].replace("'", "&#39;").replace('"', '&quot;')
            
            if st.session_state.get(f"edit_{i}", False):
                html_replacement = f"<span class='fixed-highlight' title='{safe_explanation}'>{edit['correction']}</span>"
            else:
                html_replacement = f"<span class='error-highlight' title='{safe_explanation}'>{edit['original']}</span><span class='correction-preview'>[{edit['correction']}]</span>"
            
            display_text = safe_html_replace(display_text, edit['original'], html_replacement)
                
        st.markdown("### 📄 Your Live Draft")
        st.markdown(f"<div style='background-color: white; color: black; padding: 15px; border-radius: 5px; border: 1px solid #ddd; line-height: 1.8;'>{display_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    with work_col:
        render_revision_workspace()

    with chat_col:
        st.markdown("### 💬 Your Writing Coach")
        chat_container = st.container(height=400)
        
        with chat_container:
            for message in st.session_state.messages:
                msg_content = message.get("content")
                if isinstance(msg_content, str):
                    with st.chat_message(message["role"]):
                        st.markdown(msg_content)
                    
        user_query = st.chat_input("Reply to your coach...")
        if user_query:
            log_telemetry_event("SEND_CHAT_MESSAGE", user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        assistant_msg, error = call_openrouter_api(
                            st.session_state.messages, 
                            user_api_key=st.session_state.get("user_api_key")
                        )
                        if error:
                            st.error(error)
                        else:
                            reply_text = assistant_msg.get("content", "")
                            st.markdown(reply_text)
                            st.session_state.messages.append(assistant_msg)
                            auto_save_session_log()
