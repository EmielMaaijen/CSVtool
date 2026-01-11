import streamlit as st
import pandas as pd
from groq import Groq
import json

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Universal Wealth Agent", layout="wide")

# Top 50 Dutch Merchant Dictionary (Immediate & Free)
DUTCH_MERCHANTS = {
    "ALBERT HEIJN": "Needs (Food)", "JUMBO": "Needs (Food)", "LIDL": "Needs (Food)",
    "NS-GROEP": "Needs (Transport)", "SHELL": "Needs (Transport)", "ESSO": "Needs (Transport)",
    "BOL.COM": "Wants (Shopping)", "COOLBLUE": "Wants (Shopping)", "AMAZON": "Wants (Shopping)",
    "DEGIRO": "Capital Growth", "MEESMAN": "Capital Growth", "TIKKIE": "Peer Transfer",
    "NETFLIX": "Wants (Sub)", "SPOTIFY": "Wants (Sub)", "THUISBEZORGD": "Wants (Dining)"
}

# --- 2. THE SELF-HEALING MAPPER ---
def get_universal_df(df):
    """
    Correctly maps Dutch bank headers to a standard format.
    Prevents the 'IBAN-as-Date' shift seen in Rabobank exports.
    """
    # Common Dutch headers found in Rabo, ING, and ABN
    date_options = ['Datum', 'Transactiedatum', 'Date']
    amount_options = ['Bedrag', 'Transactiebedrag', 'Amount', 'Bedrag (EUR)']
    desc_options = ['Naam tegenpartij', 'Naam / Omschrijving', 'Omschrijving', 'Description']

    # Find the right columns based on name match
    found_date = next((c for c in df.columns if c in date_options), df.columns[0])
    found_amount = next((c for c in df.columns if c in amount_options), df.columns[2] if len(df.columns) > 2 else df.columns[1])
    found_desc = next((c for c in df.columns if c in desc_options), df.columns[1])

    clean_df = df[[found_date, found_desc, found_amount]].copy()
    clean_df.columns = ['Date', 'Description', 'Amount']
    return clean_df

# --- 3. BATCH CATEGORIZATION ENGINE ---
def process_batches(client, df):
    results = []
    batch_size = 15 # Optimal for speed
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Update Progress
        progress = (i + batch_size) / len(df)
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Analyzing batch {i//batch_size + 1}...")

        # Prepare batch for AI
        batch_json = batch.to_json(orient="records")
        prompt = f"Categorize these Dutch transactions into [Needs, Wants, Growth, Income]: {batch_json}. Return ONLY a JSON list: [{{'Category': '...'}}]"

        try:
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            ai_results = json.loads(chat.choices[0].message.content).get("transactions", [])
            
            for idx, row in enumerate(batch.to_dict('records')):
                # Check local dictionary first for 100% accuracy
                local_cat = next((cat for m, cat in DUTCH_MERCHANTS.items() if m in str(row['Description']).upper()), None)
                final_cat = local_cat if local_cat else ai_results[idx].get("Category", "Lifestyle")
                results.append({**row, "Category": final_cat})
        except:
            for row in batch.to_dict('records'):
                results.append({**row, "Category": "Manual Check Required"})

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# --- 4. FRONTEND ---
st.title("🏦 Universal Bank Categorizer (2026 Edition)")

if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Add your GROQ_API_KEY to Secrets first!")
    st.stop()

file = st.file_uploader("Upload Bank CSV", type="csv")

if file:
    raw_df = pd.read_csv(file)
    with st.expander("Step 1: Raw Data Check"):
        st.dataframe(raw_df.head(5), width="stretch") # Fixed 2026 warning

    if st.button("🚀 Run Universal Categorizer"):
        # Step 1: Map columns correctly
        mapped_df = get_universal_df(raw_df)
        
        # Step 2: Batch process for speed
        final_results = process_batches(client, mapped_df)
        
        st.success("Analysis Complete!")
        # Use width='stretch' to avoid the 2026 warning
        st.dataframe(final_results, width="stretch") 
        
        csv = final_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Final Report", csv, "wealth_report.csv")
