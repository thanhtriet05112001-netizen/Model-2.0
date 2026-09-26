import streamlit as st
from openai import OpenAI
import base64
import re
import csv
import os
from datetime import datetime, timedelta
import time

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Companion in Writing", layout="wide")

# Configure OpenRouter client
api_key = st.secrets.get("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

LOG_FILE = "interaction_logs.csv"

# Timezone adjustment for Vietnam (UTC+7)
def get_vn_time():
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")

# Ensure CSV log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Exam Type", "Task Name", "Word Count", "WPM", "Estimated Pauses (sec)", "Lexical Diversity", "Acceptance Rate", "Full Draft"])

def save_interaction_csv(exam_type, task_name, word_count, wpm, pauses, ttr, acceptance_rate, draft):
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            get_vn_time(),
            exam_type,
            task_name,
            word_count,
            round(wpm, 1),
            round(pauses, 1),
            round(ttr, 2),
            f"{acceptance_rate}%",
            draft  # Full draft included here
        ])

# ==========================================
# 2. RUBRICS & METRICS
# ==========================================
def get_rubric_criteria(exam_type, task_name):
    # Detailed grading rubrics injected directly into the AI prompt
    if "IELTS" in exam_type:
        if "Task 1" in task_name:
            return """IELTS Task 1 Rubric (Band 1-9):
            1. Task Achievement: Does it highlight key features and provide a clear overview?
            2. Coherence & Cohesion: Is information logically organized with appropriate linking devices?
            3. Lexical Resource: Is there a range of academic vocabulary and awareness of collocation?
            4. Grammatical Range & Accuracy: Are there varied complex structures and error-free sentences?"""
        else:
            return """IELTS Task 2 Rubric (Band 1-9):
            1. Task Response: Is the prompt fully addressed with a clear position and supported arguments?
            2. Coherence & Cohesion: Are paragraphs well-structured with logical progression?
            3. Lexical Resource: Is there precise, varied vocabulary with minimal spelling/word formation errors?
            4. Grammatical Range & Accuracy: Is there a mix of simple and complex sentence forms with accurate punctuation?"""
    elif "TOEFL" in exam_type:
        return """TOEFL Writing Rubric (Score 0-30):
        Evaluate based on Development (explanations/details), Organization (unity and progression), and Language Use (syntactic variety, word choice, and idiomatic phrasing)."""
    else:
        return """General Academic & CEFR Rubric (A1-C2):
        Evaluate based on thesis clarity, structural organization, vocabulary precision, academic tone, and mechanical accuracy (grammar, spelling, punctuation)."""

def calculate_metrics(text, start_time):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    elapsed_time = max(time.time() - start_time, 1)
    wpm = (word_count / elapsed_time) * 60
    estimated_pauses = max(0, elapsed_time - (word_count * 0.4))
    
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences) if sentences else 1
    
    unique_words = set(words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    
    return word_count, sentence_count, ttr, wpm, estimated_pauses

# ==========================================
# 3. AI GENERATION FUNCTIONS
# ==========================================
def get_ai_feedback(exam_type, task_name, task_prompt, student_text, image_base64=None):
    rubric = get_rubric_criteria(exam_type, task_name)
    content_payload = [{"type": "text", "text": f"""
You are an expert academic writing assessor. Review the student text.
- Exam Type: {exam_type}
- Task/Module: {task_name}
- Task Prompt: {task_prompt}
- Strict Grading Rubric: {rubric}

Structure your feedback clearly:
1. **Estimated Score**: Provide the exact band/score based on the provided rubric.
2. **Analytical Breakdown**: Briefly assess the text against each of the 4 rubric criteria.
3. **Targeted Errors**: Identify 2-3 specific grammar/structural errors, quote the original sentence, explain why it is wrong, and how to fix it.
4. **Actionable Advice**: Provide 2 next steps for the student to improve. Do NOT rewrite the essay for them.

Student Text:
{student_text}
"""}]
    
    if image_base64:
        content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    response = client.chat.completions.create(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": content_payload}])
    return response.choices[0].message.content

def get_grammar_correction(student_text):
    prompt = f"""
You are a precise proofreader. Provide a mechanically corrected version of the student's text below. 
Rule 1: Fix ONLY grammar, spelling, punctuation, and awkward phrasing. 
Rule 2: Do NOT add new ideas, change the student's meaning, or write new paragraphs.
Rule 3: Output ONLY the corrected text.

Student Text:
{student_text}
"""
    response = client.chat.completions.create(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# ==========================================
# 4. UI LAYOUT: SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## Companion in Writing")
    st.caption("Elevate your academic writing.")
    st.markdown("---")
    
    st.subheader("Assessment Setup")
    exam_type = st.selectbox("Exam Type", ["IELTS", "TOEFL", "CEFR General", "Academic Writing"])
    task_name = st.selectbox("Task / Module", ["Task 1", "Task 2", "Essay", "Letter / Report"])
    
    st.markdown("---")
    evaluate_button = st.button("Generate Feedback", type="primary", use_container_width=True)
    
    st.markdown("---")
    with st.expander("Admin Log Export"):
        admin_pass = st.text_input("Admin Password", type="password")
        expected_pass = st.secrets.get("ADMIN_PASSWORD", "mysecretpassword123")
        
        if admin_pass == expected_pass:
            st.success("Authenticated")
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as f:
                    st.download_button(label="Download CSV Logs", data=f, file_name="interaction_logs.csv", mime="text/csv")
            else:
                st.info("No logs recorded yet.")
        elif admin_pass:
            st.error("Incorrect password.")

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# ==========================================
# 5. UI LAYOUT: MAIN WORKSPACE
# ==========================================
st.title("Companion in Writing")
st.markdown("Your dedicated workspace for thoughtful writing analysis, targeted scoring, and interactive revision.")

with st.container(border=True):
    task_prompt = st.text_area("Prompt / Context (Optional)", placeholder='e.g., "Write an essay analyzing the chart data..."', height=68)
    uploaded_image = st.file_uploader("Attach Chart / Graph (Optional)", type=["jpg", "jpeg", "png", "webp"])
    image_bytes_base64 = base64.b64encode(uploaded_image.read()).decode("utf-8") if uploaded_image else None

with st.container(border=True):
    student_text = st.text_area("Student Draft", placeholder="Paste writing here...", height=200)

# ==========================================
# 6. EVALUATION & INTERACTIVE TABS
# ==========================================
if evaluate_button:
    if not api_key:
        st.error("Missing OpenRouter API Key in Streamlit Secrets.")
    elif not student_text.strip():
        st.warning("Please enter a draft before requesting feedback.")
    else:
        with st.spinner("Analyzing text against rubrics..."):
            words, sentences, ttr, wpm, pauses = calculate_metrics(student_text, st.session_state.start_time)
            
            st.session_state.last_student_text = student_text
            st.session_state.last_feedback = get_ai_feedback(exam_type, task_name, task_prompt, student_text, image_bytes_base64)
            st.session_state.metrics = (words, wpm, pauses, ttr)

if "last_feedback" in st.session_state:
    st.markdown("---")
    
    # Create interactive tabs for a better user experience
    tab1, tab2, tab3 = st.tabs(["📊 Evaluation & Scoring", "🔄 Before & After (Mechanics)", "💬 Interactive Writing Coach"])
    
    # TAB 1: Main Feedback
    with tab1:
        words, wpm, pauses, ttr = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Word Count", words)
        col2.metric("Est. WPM", f"{wpm:.1f}")
        col3.metric("Est. Pauses", f"{pauses:.1f}s")
        col4.metric("Lexical Diversity", f"{ttr:.2f}")
        
        st.markdown("### Assessor Feedback")
        st.write(st.session_state.last_feedback)
        
        st.markdown("#### Revision Checklist")
        chk1 = st.checkbox("I have reviewed the structural feedback.")
        chk2 = st.checkbox("I understand the specific grammatical errors identified.")
        acceptance_rate = 100 if (chk1 and chk2) else (50 if (chk1 or chk2) else 0)
        
        # Save to CSV using the updated Vietnamese time and full draft
        save_interaction_csv(exam_type, task_name, words, wpm, pauses, ttr, acceptance_rate, st.session_state.last_student_text)

    # TAB 2: Side-by-Side Comparison
    with tab2:
        st.markdown("### Grammar & Mechanics Review")
        st.info("Compare your original draft with a mechanically corrected version. **Note:** This version only fixes grammar, vocabulary, and punctuation. It does not rewrite your ideas.")
        
        if st.button("Generate Corrected Version"):
            with st.spinner("Applying grammatical corrections..."):
                st.session_state.corrected_text = get_grammar_correction(st.session_state.last_student_text)
                
        if "corrected_text" in st.session_state:
            col_orig, col_corr = st.columns(2)
            with col_orig:
                st.subheader("Your Original Draft")
                st.write(st.session_state.last_student_text)
            with col_corr:
                st.subheader("Corrected Version")
                st.write(st.session_state.corrected_text)

    # TAB 3: Interactive Chatbot
    with tab3:
        st.markdown("### Discuss Your Writing")
        st.warning("**Learning Policy:** The writing coach will help you understand your errors, explain grammar rules, and brainstorm vocabulary. It will **not** write the essay for you.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("Ask how to fix a specific sentence, or request a grammar explanation...")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_prompt = f"""
You are an educational writing coach. 
Student's Draft: {st.session_state.last_student_text}
Previous Feedback: {st.session_state.last_feedback}

Student Query: {user_query}

Rule: Answer clearly and interactively. If they ask how to fix a specific sentence, guide them through the correction process. Do NOT write full essays for them.
"""
                    response = client.chat.completions.create(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": chat_prompt}])
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
