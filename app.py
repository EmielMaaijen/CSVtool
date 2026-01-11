import streamlit as st
import pandas as pd
from groq import Groq

# 1. Page Config
st.set_page_config(page_title="Universal Bank Agent", layout="wide")
st.title("🏦 Universal Bank Categorizer (Live Demo)")

# 2. Access your Secret API Key (We will set this up in Step 3)
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Please add your GROQ_API_KEY to the Streamlit Secrets!")
    st.stop()

# 3. The Tool Logic
uploaded_file = st.file_uploader("Upload any Bank CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Statement View", df.head(3))
    
    if st.button("🚀 Categorize with Cloud AI"):
        sample = df.head(10).to_json()
        
        # We ask the Cloud AI to do the heavy lifting
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial expert. Clean and categorize these transactions into: Needs, Wants, and Savings. Return a clean table."},
                {"role": "user", "content": f"Transactions: {sample}"}
            ],
            model="llama-3.3-70b-versatile",
        )
        
        st.success("Analysis Complete!")
        st.write(chat_completion.choices[0].message.content)