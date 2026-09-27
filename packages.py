import streamlit as st
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
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="C.O.W", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .error-highlight { background-color: #ffe6e6; color: #b30000; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ff9999; cursor: help; text-decoration: line-through; }
    .correction-preview { color: #cc0000; font-size: 0.9em; margin-left: 4px; font-weight: bold; }
    .fixed-highlight { background-color: #e6ffe6; color: #006600; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #99cc99; cursor: help; }
</style>
""", unsafe_allow_html=True)

vn_tz = timezone(timedelta(hours=7))
DEFAULT_MODEL = "google/gemma-4-31b-it:free"

LOG_FILE = "interaction_logs.json"

# ==========================================
# 2. OPENROUTER REQUEST HELPER (MATCHING CURL)
# ==========================================
def call_openrouter_api(messages, user_api_key=None, is_eval=False):
    # Resolve API Key: User Input > Environment Variable > Streamlit Secrets
    api_key = (user_api_key and user_api_key.strip()) or os.environ.get("OPENROUTER_API_KEY") or (
        st.secrets.get("OPENROUTER_API_KEY") if os.path.exists(".streamlit/secrets.toml") else None
    )
    
    if not api_key:
        return None, "🔑 No API key provided. Please enter your OpenRouter key in the sidebar."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Clean payload matching the direct working curl request
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages
    }

    # Enable reasoning for interactive chat turns, omit during structural JSON evaluations
    if not is_eval:
        payload["reasoning"] = {"enabled": True}

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
            error_msg = res_data.get("error", {}).get("message", "Unknown OpenRouter API error")
            return None, f"API Error: {error_msg}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# ==========================================
# 3. DATA LOGGING & METRICS
# ==========================================
def save_interaction_json(task_name, word_count, sent_count, wpm, ttr, accepted_edits, total_edits, chat_count, interaction_details, draft):
    log_entry = {
        "timestamp_vn": datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S"),
        "task_name": task_name,
        "word_count": word_count,
        "sentence_count": sent_count,
        "wpm": round(wpm, 1),
        "lexical_diversity": round(ttr, 2),
        "accepted_edits_count": accepted_edits,
        "total_edits_count": total_edits,
        "chat_messages_sent": chat_count,
        "accepted_edit_details": interaction_details,
        "full_draft": draft
    }
    
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
            
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
# 4. AI EVALUATION FUNCTION
# ==========================================
def get_ai_evaluation(user_key, task_name, task_prompt, student_text, image_base64=None):
    if task_name == "IELTS Task 1 (Academic)":
        rubric = "Grade based on: Task Achievement, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy."
    else:
        rubric = "Grade based on: Task Response, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy."

    prompt_content = f"""
    You are 'Companion in Writing', an expert IELTS examiner and proactive writing coach.
    Task: {task_name}
    Prompt: {task_prompt}
    Rubric: {rubric}
    Student Text: {student_text}

    IMPORTANT: Return ONLY a valid raw JSON object. Do not wrap it in markdown code blocks, do not output any introductory or concluding text.

    Required JSON Structure:
    {{
        "band_score": "Estimated IELTS Band (e.g., 6.5)",
        "overall_feedback": "A short, encouraging paragraph summarizing strengths and weaknesses.",
        "edits": [
            {{
                "original": "Include 2 to 4 words from the text to provide distinct context (e.g., 'make an mistake' instead of 'an'). MUST match the original text exactly.",
                "correction": "The corrected phrase",
                "explanation": "Brief explanation of the grammar/spelling rule"
            }}
        ],
        "coach_opening_chat": "A friendly question asking the student about a specific error you noticed, inviting them to discuss it."
    }}
    """
    
    user_payload = [{"type": "text", "text": prompt_content}]
    if image_base64:
        user_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    messages = [{"role": "user", "content": user_payload}]
    
    # Call with is_eval=True to disable reasoning conflicts during JSON evaluation
    message_data, error = call_openrouter_api(messages, user_api_key=user_key, is_eval=True)
    if error:
        st.error(error)
        return None

    raw_content = message_data.get("content", "{}")
    
    # Safely extract { ... } JSON block using regex
    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    cleaned_content = json_match.group(0) if json_match else raw_content

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        st.error("Failed to parse evaluation response into JSON. Please click Submit again.")
        return None

# ==========================================
# 5. UI LAYOUT & SIDEBAR
# ==========================================
st.title("C.O.W")
st.markdown("Your interactive IELTS workspace. Draft, review, and collaborate with your AI coach.")

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# Sidebar for User API Key
with st.sidebar:
    st.subheader("🔑 API Key Setup")
    user_api_key = st.text_input(
        "OpenRouter Key (Optional)", 
        type="password", 
        help="Get a free key at https://openrouter.ai/keys"
    )
    st.session_state.user_api_key = user_api_key
    if not user_api_key:
        st.caption("ℹ️ Using default server key.")

# Admin Tools Popover
with st.popover("⚙️ Admin Tools"):
    admin_pass = st.text_input("Password", type="password")
    expected_admin_pass = os.environ.get("ADMIN_PASSWORD") or (
        st.secrets.get("ADMIN_PASSWORD") if os.path.exists(".streamlit/secrets.toml") else "secret123"
    )
    if admin_pass == expected_admin_pass:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button(
                    label="📥 Download Full JSON Logs", 
                    data=f, 
                    file_name=f"writing_logs_{datetime.now(vn_tz).strftime('%Y%m%d')}.json", 
                    mime="application/json"
                )

# Drafting Section
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
            with st.spinner(f"Analyzing with OpenRouter ({DEFAULT_MODEL})..."):
                image_b64 = base64.b64encode(uploaded_image.read()).decode("utf-8") if uploaded_image else None
                eval_data = get_ai_evaluation(st.session_state.get("user_api_key"), task_name, task_prompt, student_text, image_b64)
                
                if eval_data:
                    st.session_state.eval_data = eval_data
                    st.session_state.original_text = student_text
                    st.session_state.evaluated = True
                    st.session_state.messages = [{"role": "assistant", "content": eval_data["coach_opening_chat"]}]
                    
                    for i in range(len(eval_data.get("edits", []))):
                        st.session_state[f"edit_{i}"] = False
                    
                    words, sents, wpm, ttr = calculate_nltk_metrics(student_text, st.session_state.start_time)
                    st.session_state.metrics = {"words": words, "sents": sents, "wpm": wpm, "ttr": ttr}
                    st.rerun()

# ==========================================
# 6. INTERACTIVE WORKSPACE
# ==========================================
if st.session_state.get("evaluated", False):
    eval_data = st.session_state.eval_data
    
    st.markdown("---")
    st.subheader(f"🏆 Estimated Score: **{eval_data.get('band_score', 'N/A')}**")
    st.info(eval_data.get('overall_feedback', ''))
    
    work_col, chat_col = st.columns([1.5, 1])
    
    with work_col:
        st.markdown("### 🔍 Interactive Revisions")
        st.caption("Check the box to accept a correction. Hover over highlighted text in the draft below for explanations.")
        
        accepted_count = 0
        accepted_log_details = []
        
        edits_list = eval_data.get("edits", [])
        sorted_edits = sorted(edits_list, key=lambda x: len(x['original']), reverse=True)
        
        for i, edit in enumerate(sorted_edits):
            is_accepted = st.checkbox(f"**Fix:** {edit['original']} ➔ {edit['correction']}", key=f"edit_{i}")
            if is_accepted:
                accepted_count += 1
                accepted_log_details.append(f"Fixed: {edit['original']}->{edit['correction']}")
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

        if st.button("Save Revision Progress"):
            metrics = st.session_state.metrics
            user_chat_count = len([m for m in st.session_state.messages if m.get("role") == "user"])
            interaction_str = " | ".join(accepted_log_details) if accepted_log_details else "No edits accepted"
            
            save_interaction_json(
                task_name, metrics['words'], metrics['sents'], metrics['wpm'], metrics['ttr'], 
                accepted_count, len(edits_list), user_chat_count, 
                interaction_str, st.session_state.original_text
            )
            st.success("Interaction metrics and NLTK analysis saved to JSON.")

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
            st.session_state.messages.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        assistant_msg, error = call_openrouter_api(
                            st.session_state.messages, 
                            user_api_key=st.session_state.get("user_api_key"),
                            is_eval=False
                        )
                        if error:
                            st.error(error)
                        else:
                            reply_text = assistant_msg.get("content", "")
                            st.markdown(reply_text)
                            st.session_state.messages.append(assistant_msg)
