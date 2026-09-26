import streamlit as st
from openai import OpenAI
import base64
import re
import json
import os
from datetime import datetime

# 1. Page Configuration for a Wide Layout
st.set_page_config(page_title="AI Writing Assessor", layout="wide")

# 2. Configure the OpenRouter client securely
api_key = st.secrets.get("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

LOG_FILE = "interaction_logs.json"

# 3. Function to log interactions server-side
def save_interaction(exam_type, task_name, word_count, ttr, feedback):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exam_type": exam_type,
        "task_name": task_name,
        "word_count": word_count,
        "lexical_diversity": round(ttr, 2),
        "feedback_summary": feedback[:150] + "..." # Save snippet of feedback
    }
    
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
            
    logs.append(log_entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

# 4. Pure Python Metrics Calculator
def calculate_metrics(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences) if sentences else 1
    
    unique_words = set(words)
    lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0
    
    return word_count, sentence_count, lexical_diversity

# 5. Define AI feedback function via OpenRouter
def get_ai_feedback(exam_type, task_name, task_prompt, student_text, image_base64=None):
    content_payload = [
        {
            "type": "text",
            "text": f"""
You are an expert English language assessor. Review the student text based on the following criteria:
- Exam Type: {exam_type}
- Task/Module: {task_name}
- Task Prompt / Context: {task_prompt}

Please evaluate the student writing and provide:
1. Estimate the CEFR level (A1 to C2) and approximate score (e.g., IELTS Band).
2. Identify 2-3 specific grammar or structural errors.
3. Provide constructive feedback on how to improve coherence, task response, and vocabulary.

Student Text:
{student_text}
"""
        }
    ]
    
    if image_base64:
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": content_payload}]
    )
    return response.choices[0].message.content

# ==========================================
# UI LAYOUT: SIDEBAR & MAIN CONTAINER
# ==========================================

with st.sidebar:
    st.subheader("⚙️ Assessment Setup")
    exam_type = st.selectbox("Exam Type", ["IELTS", "TOEFL", "CEFR General", "Academic Writing"])
    task_name = st.selectbox("Task / Module", ["Task 1", "Task 2", "Essay", "Letter / Report"])
    
    st.markdown("---")
    evaluate_button = st.button("Get Feedback", type="primary", use_container_width=True)
    
    # --- PRIVATE ADMIN PANEL FOR LOGS ---
    st.markdown("---")
    with st.expander("🔒 Admin Log Download"):
        admin_pass = st.text_input("Admin Password", type="password")
        # Set your password in Streamlit secrets or hardcode a check here securely
        expected_pass = st.secrets.get("ADMIN_PASSWORD", "mysecretpassword123")
        
        if admin_pass == expected_pass:
            st.success("Authenticated!")
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as f:
                    st.download_button(
                        label="📥 Download JSON Logs",
                        data=f,
                        file_name="interaction_logs.json",
                        mime="application/json"
                    )
            else:
                st.info("No interaction logs recorded yet.")
        elif admin_pass:
            st.error("Incorrect password.")

# Main Page Header
st.title("📝 AI Writing Assessor")
st.markdown("Configure your prompt, upload charts or reference materials, and evaluate student submissions.")

# Section 1: Task Prompt & Image Upload
with st.container(border=True):
    st.markdown("### 📋 Task Prompt *(Optional)*")
    task_prompt = st.text_area("Prompt Context", placeholder='e.g., "Write an essay analyzing the impact of technology..."', height=100, label_visibility="collapsed")
    
    st.markdown("### 📊 Upload Chart / Graph *(Optional)*")
    uploaded_image = st.file_uploader("Choose an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    image_bytes_base64 = None
    if uploaded_image is not None:
        image_bytes_base64 = base64.b64encode(uploaded_image.read()).decode("utf-8")
        st.success("Image successfully uploaded and attached.")

# Section 2: Student Text Input
with st.container(border=True):
    st.markdown("### ✍️ Student Writing")
    student_text = st.text_area("Student Text", placeholder="Paste student writing here...", height=220, label_visibility="collapsed")

# ==========================================
# EVALUATION TRIGGER
# ==========================================
if evaluate_button:
    if not api_key:
        st.error("Missing OpenRouter API Key. Please add it to your Streamlit Secrets.")
    elif not student_text.strip():
        st.warning("Please enter student text before requesting feedback.")
    else:
        with st.spinner("Analyzing linguistic structures and generating AI feedback..."):
            # 1. Calculate Metrics
            words, sentences, ttr = calculate_metrics(student_text)
            
            # 2. Display Metrics Dashboard
            st.markdown("---")
            st.subheader("📊 Linguistic Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Word Count", words)
            col2.metric("Sentence Count", sentences)
            col3.metric("Lexical Diversity (TTR)", f"{ttr:.2f}")
            
            # 3. Display AI Evaluation
            st.subheader("🤖 Qualitative AI Feedback")
            try:
                ai_feedback = get_ai_feedback(
                    exam_type=exam_type,
                    task_name=task_name,
                    task_prompt=task_prompt,
                    student_text=student_text,
                    image_base64=image_bytes_base64
                )
                st.markdown(ai_feedback)
                
                # 4. Save interaction log quietly in the background
                save_interaction(exam_type, task_name, words, ttr, ai_feedback)
                
            except Exception as e:
                st.error(f"Error connecting to OpenRouter API: {e}")
