import streamlit as st
from openai import OpenAI
import base64
import re
import csv
import os
import json
import time
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Companion in Writing", layout="wide", initial_sidebar_state="collapsed")

# Modern Styling for Interactive Hover Highlights
st.markdown("""
<style>
    /* Unaccepted Error: Red strikethrough with hover cursor */
    .error-highlight { background-color: #ffe6e6; color: #b30000; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ff9999; cursor: help; text-decoration: line-through; }
    /* Preview of the correction next to the error */
    .correction-preview { color: #cc0000; font-size: 0.9em; margin-left: 4px; font-weight: bold; }
    /* Accepted Fix: Green styling with hover cursor */
    .fixed-highlight { background-color: #e6ffe6; color: #006600; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #99cc99; cursor: help; }
</style>
""", unsafe_allow_html=True)

vn_tz = timezone(timedelta(hours=7))

api_key = st.secrets.get("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

LOG_FILE = "interaction_logs.csv"

# ==========================================
# 2. DATA LOGGING & METRICS
# ==========================================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp_VN", "Task Name", "Word Count", "WPM", "Lexical Diversity", "Accepted Edits", "Total Edits", "Chat Messages Sent", "Accepted Edit Details", "Full Draft"])

def save_interaction_csv(task_name, word_count, wpm, ttr, accepted_edits, total_edits, chat_count, interaction_details, draft):
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S"),
            task_name,
            word_count,
            round(wpm, 1),
            round(ttr, 2),
            accepted_edits,
            total_edits,
            chat_count,
            interaction_details,
            draft.replace("\n", " | ")
        ])

def calculate_metrics(text, start_time):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    elapsed_time = max(time.time() - start_time, 1)
    wpm = (word_count / elapsed_time) * 60
    unique_words = set(words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    return word_count, wpm, ttr

# ==========================================
# 3. AI PROMPT & RUBRICS
# ==========================================
def get_ai_evaluation(task_name, task_prompt, student_text, image_base64=None):
    if task_name == "IELTS Task 1 (Academic)":
        rubric = "Grade based on: Task Achievement (accurate data overview and comparisons), Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy."
    else:
        rubric = "Grade based on: Task Response (clear position, developed arguments), Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy."

    prompt_content = f"""
    You are 'Companion in Writing', an expert IELTS examiner and proactive writing coach.
    Task: {task_name}
    Prompt: {task_prompt}
    Rubric: {rubric}
    Student Text: {student_text}

    Provide your response STRICTLY as a valid JSON object with the following exact keys:
    {{
        "band_score": "Estimated IELTS Band (e.g., 6.5)",
        "overall_feedback": "A short, encouraging paragraph summarizing strengths and weaknesses.",
        "edits": [
            {{
                "original": "The EXACT misspelled word or grammatically incorrect phrase from the text (MUST match the text exactly, case-sensitive)",
                "correction": "The corrected version",
                "explanation": "Brief explanation of the grammar/spelling rule"
            }}
        ],
        "coach_opening_chat": "A friendly, proactive question asking the student about a specific error you noticed, inviting them to discuss it."
    }}
    """
    
    payload = [{"type": "text", "text": prompt_content}]
    if image_base64:
        payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": payload}]
    )
    return json.loads(response.choices[0].message.content)

# ==========================================
# 4. UI: COMPACT HEADER & SETUP
# ==========================================
st.title("✨ Companion in Writing")
st.markdown("Your interactive IELTS workspace. Draft, review, and collaborate with your AI coach.")

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

with st.popover("⚙️ Admin Tools"):
    admin_pass = st.text_input("Password", type="password")
    if admin_pass == st.secrets.get("ADMIN_PASSWORD", "secret123"):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button("📥 Download Full CSV", f, file_name=f"writing_logs_{datetime.now(vn_tz).strftime('%Y%m%d')}.csv", mime="text/csv")

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
            with st.spinner("Analyzing against IELTS rubrics..."):
                image_b64 = base64.b64encode(uploaded_image.read()).decode("utf-8") if uploaded_image else None
                eval_data = get_ai_evaluation(task_name, task_prompt, student_text, image_b64)
                
                st.session_state.eval_data = eval_data
                st.session_state.original_text = student_text
                st.session_state.evaluated = True
                st.session_state.messages = [{"role": "assistant", "content": eval_data["coach_opening_chat"]}]
                
                for i in range(len(eval_data["edits"])):
                    st.session_state[f"edit_{i}"] = False
                
                words, wpm, ttr = calculate_metrics(student_text, st.session_state.start_time)
                st.session_state.metrics = {"words": words, "wpm": wpm, "ttr": ttr}
                st.rerun()

# ==========================================
# 5. INTERACTIVE WORKSPACE
# ==========================================
if st.session_state.get("evaluated", False):
    eval_data = st.session_state.eval_data
    
    st.markdown("---")
    st.subheader(f"🏆 Estimated Score: **{eval_data['band_score']}**")
    st.info(eval_data['overall_feedback'])
    
    work_col, chat_col = st.columns([1.5, 1])
    
    with work_col:
        st.markdown("### 🔍 Interactive Revisions")
        st.caption("Check the box to accept a correction. Hover over highlighted text in the draft below for explanations.")
        
        accepted_count = 0
        accepted_log_details = []
        
        # Sort edits by length descending so longer phrases are replaced before single words
        sorted_edits = sorted(eval_data["edits"], key=lambda x: len(x['original']), reverse=True)
        
        for i, edit in enumerate(sorted_edits):
            is_accepted = st.checkbox(f"**Fix:** {edit['original']} ➔ {edit['correction']}", key=f"edit_{i}")
            if is_accepted:
                accepted_count += 1
                accepted_log_details.append(f"Fixed: {edit['original']}->{edit['correction']}")
            st.caption(f"*Why?* {edit['explanation']}")
        
        # Generate the interactive Live Draft
        display_text = st.session_state.original_text
        for i, edit in enumerate(sorted_edits):
            safe_explanation = edit['explanation'].replace("'", "&#39;") # Escape quotes for HTML tooltips
            
            if st.session_state.get(f"edit_{i}", False):
                # Accepted: Show correction in green, explanation on hover
                html_replacement = f"<span class='fixed-highlight' title='{safe_explanation}'>{edit['correction']}</span>"
                display_text = display_text.replace(edit['original'], html_replacement)
            else:
                # Not accepted: Show original with red strikethrough + correction preview, explanation on hover
                html_replacement = f"<span class='error-highlight' title='{safe_explanation}'>{edit['original']}</span><span class='correction-preview'>[{edit['correction']}]</span>"
                display_text = display_text.replace(edit['original'], html_replacement)
                
        st.markdown("### 📄 Your Live Draft")
        st.markdown(f"<div style='background-color: white; color: black; padding: 15px; border-radius: 5px; border: 1px solid #ddd; line-height: 1.8;'>{display_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

        if st.button("Save Revision Progress"):
            metrics = st.session_state.metrics
            user_chat_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            interaction_str = " | ".join(accepted_log_details) if accepted_log_details else "No edits accepted"
            
            save_interaction_csv(
                task_name, metrics['words'], metrics['wpm'], metrics['ttr'], 
                accepted_count, len(eval_data["edits"]), user_chat_count, 
                interaction_str, st.session_state.original_text
            )
            st.success("Interaction metrics and revisions saved securely.")

    with chat_col:
        st.markdown("### 💬 Your Writing Coach")
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
        user_query = st.chat_input("Reply to your coach...")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.chat_message("assistant"):
                    with st.spinner("Typing..."):
                        chat_prompt = f"""
                        You are the 'Companion in Writing' IELTS coach.
                        Original draft: {st.session_state.original_text}
                        Feedback given: {eval_data['overall_feedback']}
                        Student query: {user_query}
                        Rule: Do not rewrite paragraphs for them. Explain grammar pedagogically.
                        """
                        response = client.chat.completions.create(
                            model="openai/gpt-4o-mini",
                            messages=[{"role": "user", "content": chat_prompt}]
                        )
                        reply = response.choices[0].message.content
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
