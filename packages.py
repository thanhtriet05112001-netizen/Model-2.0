import streamlit as st
from openai import OpenAI
import base64
import re
import csv
import io
import os
from datetime import datetime
import time

# 1. Page Configuration
st.set_page_config(page_title="C.O.W - Companion in Writing", layout="wide")

# 2. Configure OpenRouter client securely
api_key = st.secrets.get("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

LOG_FILE = "interaction_logs.csv"

# Ensure CSV log file exists with headers
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Exam Type", "Task Name", "Word Count", "WPM", "Estimated Pauses (sec)", "Lexical Diversity", "Acceptance Rate", "First Draft Snippet"])

def save_interaction_csv(exam_type, task_name, word_count, wpm, pauses, ttr, acceptance_rate, draft):
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exam_type,
            task_name,
            word_count,
            round(wpm, 1),
            round(pauses, 1),
            round(ttr, 2),
            f"{acceptance_rate}%",
            draft[:100].replace("\n", " ")
        ])

# 3. Dynamic Rubrics based on Writing Type
def get_rubric_criteria(exam_type, task_name):
    if "IELTS" in exam_type:
        if "Task 1" in task_name:
            return "IELTS Task 1 Academic/General criteria: Focus strictly on Task Achievement (overview, data selection), Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy."
        else:
            return "IELTS Task 2 criteria: Focus strictly on Task Response (position, arguments), Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy."
    elif "TOEFL" in exam_type:
        return "TOEFL Writing criteria: Focus on development, organization, unity, progression, and syntactic/lexical precision."
    else:
        return "General Academic Writing criteria: Focus on clarity, thesis strength, structural organization, vocabulary precision, and mechanics."

# 4. Pure Python Metrics & Fluency Calculator
def calculate_metrics(text, start_time):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    elapsed_time = max(time.time() - start_time, 1) # seconds
    wpm = (word_count / elapsed_time) * 60
    
    # Rough pause approximation based on time vs length
    estimated_pauses = max(0, elapsed_time - (word_count * 0.4))
    
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences) if sentences else 1
    
    unique_words = set(words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    
    return word_count, sentence_count, ttr, wpm, estimated_pauses

# 5. AI Feedback Function with Targeted Rubric
def get_ai_feedback(exam_type, task_name, task_prompt, student_text, image_base64=None):
    rubric = get_rubric_criteria(exam_type, task_name)
    content_payload = [
        {
            "type": "text",
            "text": f"""
You are C.O.W (Companion in Writing), an expert AI writing tutor. Review the student text.
- Exam Type: {exam_type}
- Task/Module: {task_name}
- Task Prompt: {task_prompt}
- Evaluation Rubric: {rubric}

Provide your feedback structured cleanly with:
1. **Band/Score Estimate**: Estimated score based on the rubric.
2. **Key Strengths**: What was done well.
3. **Targeted Errors & Recommendations**: Identify 2-3 specific grammatical or structural errors, explaining why they are errors and how to fix them.
4. **Actionable Advice**: Clear guidance for improvement (DO NOT rewrite the essay for them).

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
    st.title("🐮 C.O.W")
    st.caption("Companion in Writing")
    st.markdown("---")
    
    st.subheader("⚙️ Assessment Setup")
    exam_type = st.selectbox("Exam Type", ["IELTS", "TOEFL", "CEFR General", "Academic Writing"])
    task_name = st.selectbox("Task / Module", ["Task 1", "Task 2", "Essay", "Letter / Report"])
    
    st.markdown("---")
    evaluate_button = st.button("Get Feedback", type="primary", use_container_width=True)
    
    # --- PRIVATE ADMIN PANEL (CSV DOWNLOAD) ---
    st.markdown("---")
    with st.expander("🔒 Admin Log Download"):
        admin_pass = st.text_input("Admin Password", type="password")
        expected_pass = st.secrets.get("ADMIN_PASSWORD", "mysecretpassword123")
        
        if admin_pass == expected_pass:
            st.success("Authenticated!")
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as f:
                    st.download_button(
                        label="📥 Download CSV Logs",
                        data=f,
                        file_name="interaction_logs.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No logs recorded yet.")
        elif admin_pass:
            st.error("Incorrect password.")

# Track start time for typing speed (WPM / Pauses)
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# Main Page Header
st.title("🐮 C.O.W: Companion in Writing")
st.markdown("Your intelligent assistant for precise writing evaluation, targeted rubric scoring, and guided revision.")

# Section 1: Task Prompt & Image Upload
with st.container(border=True):
    st.markdown("### 📋 Task Prompt *(Optional)*")
    task_prompt = st.text_area("Prompt Context", placeholder='e.g., "Write an essay analyzing the chart data..."', height=90, label_visibility="collapsed")
    
    st.markdown("### 📊 Upload Chart / Graph *(Optional)*")
    uploaded_image = st.file_uploader("Choose an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    image_bytes_base64 = None
    if uploaded_image is not None:
        image_bytes_base64 = base64.b64encode(uploaded_image.read()).decode("utf-8")
        st.success("Image successfully attached.")

# Section 2: Student Text Input
with st.container(border=True):
    st.markdown("### ✍️ Student Writing")
    student_text = st.text_area("Student Text", placeholder="Paste student writing here...", height=200, label_visibility="collapsed")

# ==========================================
# EVALUATION TRIGGER
# ==========================================
if evaluate_button:
    if not api_key:
        st.error("Missing OpenRouter API Key. Please add it to your Streamlit Secrets.")
    elif not student_text.strip():
        st.warning("Please enter student text before requesting feedback.")
    else:
        with st.spinner("Analyzing writing metrics and rubric compliance..."):
            # Calculate metrics
            words, sentences, ttr, wpm, pauses = calculate_metrics(student_text, st.session_state.start_time)
            
            # Store data in session state for interactive elements & chat
            st.session_state.last_student_text = student_text
            st.session_state.last_feedback = get_ai_feedback(exam_type, task_name, task_prompt, student_text, image_bytes_base64)
            
            # Display Metrics Dashboard
            st.markdown("---")
            st.subheader("📊 Writing & Fluency Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Word Count", words)
            col2.metric("Est. WPM", f"{wpm:.1f}")
            col3.metric("Est. Pauses", f"{pauses:.1f}s")
            col4.metric("Lexical Diversity (TTR)", f"{ttr:.2f}")
            
            # Display AI Evaluation
            st.subheader("🤖 Qualitative AI Feedback")
            st.markdown(st.session_state.last_feedback)
            
            # Interactive Recommendation Selection (Feature 4)
            st.markdown("### 🛠️ Interactive Revision Checklist")
            st.caption("Select the error corrections you want to acknowledge or integrate into your revision notes:")
            chk1 = st.checkbox("Acknowledge grammatical accuracy feedback and plan corrections")
            chk2 = st.checkbox("Acknowledge lexical enhancement suggestions")
            
            acceptance_rate = 100 if (chk1 and chk2) else (50 if (chk1 or chk2) else 0)
            
            # Save CSV log entry
            save_interaction_csv(exam_type, task_name, words, wpm, pauses, ttr, acceptance_rate, student_text)

# ==========================================
# INTERACTIVE CHAT ASSISTANT WITH DISCLAIMER (Feature 3)
# ==========================================
if "last_feedback" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Chat with C.O.W (Writing Coach)")
    st.info("💡 **Disclaimer & Policy**: C.O.W is designed to guide your learning. **The AI is strictly prohibited from writing your essay, rewriting your paragraphs in full, or doing your work for you.** Ask clarifying questions or request explanations on grammar rules instead!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about the feedback or grammar rules...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_prompt = f"""
You are C.O.W, an strict educational writing coach. 
The student's text is: {st.session_state.get('last_student_text', '')}
Previous feedback given: {st.session_state.get('last_feedback', '')}

Student query: {user_query}

CRITICAL RULE: You must NEVER write the student's essay, provide full sentences for them to copy-paste, or do their writing assignment for them. Answer their questions pedagogically, explain grammar concepts, and guide them to rewrite it themselves.
"""
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": chat_prompt}]
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
