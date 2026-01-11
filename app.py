import streamlit as st
import pandas as pd
from groq import Groq
import json
import time

# --- 1. CONFIG & MERCHANT DATA ---
st.set_page_config(page_title="Universal Bank Agent", layout="wide", page_icon="🏦")

# Top 50 Dutch Merchant Logic Table (Immediate & Free)
DUTCH_MERCHANTS = {
    "ALBERT HEIJN": "Groceries", "JUMBO": "Groceries", "LIDL": "Groceries", "ALDI": "Groceries", 
    "NS-GROEP": "Transport", "GVB": "Transport", "SHELL": "Transport",
    "BOL.COM": "Shopping", "COOLBLUE": "Shopping", "AMAZON": "Shopping",
    "DEGIRO": "Investment", "MEESMAN": "Investment", "TIKKIE": "Peer Transfer"
}

# --- 2. THE CORE LOGIC ---

def get_logic_category(desc):
    """Fast check against local merchant dictionary."""
    desc = desc.upper()
    for m, cat in DUTCH_MERCHANTS.items():
        if m in desc: return cat
    return None

def process_batch_with_ai(client, batch_df):
    """Sends 15 rows at once to the Cloud AI for fast results."""
    # Convert batch to a simple text format to save tokens
    batch_json = batch_df.to_json(orient="records")
    
    prompt = f"""
    Categorize these Dutch bank transactions into: [Fixed, Lifestyle, Investment, Income].
    Data: {batch_json}
    Return ONLY a JSON list of objects: [{{"Category": "..."}}, ...]
    Ensure the order matches the input exactly.
    """
    
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return json.loads(chat.choices[0].message.content).get("transactions", [])
    except Exception as e:
        st.error(f"AI Error: {e}")
        return [{"Category": "Error"}] * len(batch_df)

# --- 3. THE FRONTEND ---

st.title("🏦 Universal Bank Categorizer")
st.info("Logic: Batch AI Processing + Progress Tracking")

# API Key Validation
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.warning("Please add your GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload Bank CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if st.button("🚀 Start Fast Batch Processing"):
        # Progress Bar Initialization
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        batch_size = 15  # Optimal for speed vs token limits
        
        # Split data into batches for speed
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Update Progress UI
            progress = (i + batch_size) / len(df)
            status_text.text(f"Categorizing batch {i//batch_size + 1}...")
            progress_bar.progress(min(progress, 1.0))
            
            # 1. First attempt local logic
            batch_results = []
            ai_batch_list = []
            
            for _, row in batch.iterrows():
                # We assume standard columns exist after your mapping step
                desc = str(row.get('Description', row.iloc[1]))
                cat = get_logic_category(desc)
                
                if cat:
                    batch_results.append({"Date": row.iloc[0], "Desc": desc, "Amount": row.iloc[2], "Category": cat})
                else:
                    # Collect unknowns for one AI call
                    ai_batch_list.append(row)
            
            # 2. Process unknowns via AI in ONE call
            if ai_batch_list:
                ai_df = pd.DataFrame(ai_batch_list)
                ai_responses = process_batch_with_ai(client, ai_df)
                
                # Merge AI results back
                for idx, ai_row in enumerate(ai_batch_list):
                    category = ai_responses[idx].get("Category", "Unknown") if idx < len(ai_responses) else "Unknown"
                    batch_results.append({"Date": ai_row.iloc[0], "Desc": ai_row.iloc[1], "Amount": ai_row.iloc[2], "Category": category})
            
            results.extend(batch_results)

        # Final UI Update
        progress_bar.empty()
        status_text.empty()
        st.success("Successfully categorized all transactions!")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
