import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 1. CONFIGURATIE & OPTIES ---
st.set_page_config(page_title="Grootboek Agent Pro", layout="wide")
MODEL_FILE = 'trained_model.joblib'

# De officiële lijst met grootboekrekeningen voor je dropdown
GROOTBOEK_OPTIES = [
    "1000 Kruisposten",
    "1600 Crediteuren",
    "4000 Huisvestingskosten",
    "4100 Kantoorkosten",
    "4200 Verkoopkosten",
    "4300 Autokosten",
    "4400 Kantinekosten",
    "4500 Reis- en verblijfwosten",
    "7000 Inkoop",
    "8000 Omzet"
]

# --- 2. FUNCTIES ---

def standardize_df(df):
    """Zet kolommen om naar Date, Description, Amount."""
    cols = df.columns.tolist()
    date_opts = ['Datum', 'Date', 'Timestamp']
    desc_opts = ['Omschrijving', 'Description', 'Counterparty']
    amt_opts = ['Bedrag', 'Amount', 'Amount_EUR']

    f_date = next((c for c in cols if c in date_opts), cols[0])
    f_desc = next((c for c in cols if c in desc_opts), cols[1])
    f_amt = next((c for c in cols if c in amt_opts), cols[2])

    df_clean = df.rename(columns={f_date: 'Date', f_desc: 'Description', f_amt: 'Amount'})
    return df_clean[['Date', 'Description', 'Amount']]

@st.cache_resource
def get_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

# --- 3. UI LAYOUT ---
st.title("🏦 Slimme Grootboek Agent")

tab1, tab2 = st.tabs(["🧠 Stap 1: Leerproces & Correctie", "🚀 Stap 2: Voorspellen"])

with tab1:
    st.header("Data Review & Training")
    st.write("Upload je data, pas de categorieën aan waar nodig, en train de AI.")

    train_file = st.file_uploader("Upload Trainingsdata (CSV)", type="csv")

    if train_file:
        df_raw = pd.read_csv(train_file)
        df_to_review = standardize_df(df_raw)

        # Voeg een Category kolom toe als die nog niet bestaat
        if 'Category' not in df_to_review.columns:
            df_to_review['Category'] = GROOTBOEK_OPTIES[0] # Default waarde

        st.subheader("📝 Controleer en pas aan")
        st.info("Klik op de 'Category' kolom om een grootboekrekening te kiezen.")

        # De Data Editor met Dropdown
        edited_df = st.data_editor(
            df_to_review,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Grootboekrekening",
                    help="Kies de juiste rekening voor deze transactie",
                    options=GROOTBOEK_OPTIES,
                    required=True,
                )
            },
            hide_index=True,
            width="stretch"
        )

        if st.button("✅ Bevestig & Train Model"):
            with st.spinner("AI wordt getraind op jouw keuzes..."):
                X = edited_df['Description'].astype(str)
                y = edited_df['Category'].astype(str)
                
                model = get_pipeline()
                model.fit(X, y)
                joblib.dump(model, MODEL_FILE)
                st.success("Het model is getraind en klaar voor gebruik!")

with tab2:
    st.header("Nieuwe Voorspellingen")
    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ Je moet eerst het model trainen in Tab 1.")
    else:
        predict_file = st.file_uploader("Upload nieuwe bank-export", type="csv")
        if predict_file:
            df_new = pd.read_csv(predict_file)
            df_mapped = standardize_df(df_new)
            
            if st.button("🚀 Voorspel Grootboekrekeningen"):
                model = joblib.load(MODEL_FILE)
                df_mapped['Grootboekrekening'] = model.predict(df_mapped['Description'].astype(str))
                
                st.dataframe(df_mapped, width="stretch")
                
                # Export
                csv = df_mapped.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Rapport", csv, "boekhouding_klaar.csv")
