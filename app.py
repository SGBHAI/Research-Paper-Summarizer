import os
import json
import time
import streamlit as st
import pypdf
from google import genai
from google.genai import types

st.set_page_config(page_title="Academic Paper Summarizer", layout="wide")
st.title("📄 Research Paper Summarizer")

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY secret or environment variable not found.")
    st.stop()

client = genai.Client(api_key=api_key)

uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    reader = pypdf.PdfReader(uploaded_file)
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    st.info(f"Loaded {len(reader.pages)} pages ({len(raw_text)} characters).")
    
    if st.button("Summarize Paper", type="primary"):
        with st.spinner("Analyzing research text..."):
            schema = {
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "methodology": {"type": "string"},
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                    "takeaways": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["objective", "methodology", "findings", "limitations", "takeaways"]
            }

            models_to_try = ["gemma-4-31b-it", "gemini-2.5-flash", "gemini-2.0-flash"]
            response_text = None
            last_error = None

            for model_name in models_to_try:
                for attempt in range(2):
                    try:
                        res = client.models.generate_content(
                            model=model_name,
                            contents=f"Extract structured summary from this paper:\n\n{raw_text[:40000]}",
                            config=types.GenerateContentConfig(
                                system_instruction="You are an expert academic research summarizer. Extract objective, methodology, findings, limitations, and takeaways.",
                                temperature=0.2,
                                response_mime_type="application/json",
                                response_schema=schema
                            )
                        )
                        response_text = res.text
                        st.caption(f"Generated using: `{model_name}`")
                        break
                    except Exception as e:
                        last_error = e
                        time.sleep(1)
                if response_text:
                    break

            if not response_text:
                st.error(f"All model endpoints busy. Error: {last_error}")
                st.stop()

            try:
                data = json.loads(response_text)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎯 Objective")
                    st.write(data["objective"])
                    
                    st.subheader("⚙️ Methodology")
                    st.write(data["methodology"])

                with col2:
                    st.subheader("📊 Findings")
                    for item in data["findings"]:
                        st.markdown(f"- {item}")

                st.divider()

                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("⚠️ Limitations")
                    for item in data["limitations"]:
                        st.markdown(f"- {item}")

                with col4:
                    st.subheader("💡 Key Takeaways")
                    for item in data["takeaways"]:
                        st.markdown(f"- {item}")

            except Exception as parse_err:
                st.error(f"Failed to parse structured output: {parse_err}")