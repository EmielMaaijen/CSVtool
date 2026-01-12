import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 1. CONFIGURATIE & CONSTANTEN ---
st.set_page_config(page_title="Slimme Boekhoud Agent", layout="wide", page_icon="🏦")
MODEL_FILE = 'trained_model.joblib'

# --- 2. UNIVERSELE MAPPING LOGICA ---
def standardize_df(df):
    """
    Zet verschillende bankformaten (Rabo, ING, Bunq/Fintech) om naar 
    een standaard formaat: Date, Description, Amount.
    """
    cols = df.columns.tolist()
    
    # Optie A: Moderne Fintech Stijl (Counterparty + Reference)
    if 'Counterparty' in cols and 'Reference' in cols:
        df['Description'] = df['Counterparty'].fillna('') + " " + df['Reference'].fillna('')
        df['Date'] = df.get('Timestamp', df.iloc[:, 0])
        df['Amount'] = df.get('Amount_EUR', df.iloc[:, 3])
        
    # Optie B: Traditionele Nederlandse Banken (ING, Rabo, ABN)
    else:
        date_opts = ['Datum', 'Transactiedatum', 'Date']
        desc_opts = ['Naam / Omschrijving', 'Naam tegenpartij', 'Omschrijving', 'Description']
        amt_opts = ['Bedrag (EUR)', 'Bedrag', 'Amount', 'Transactiebedrag']

        f_date = next((c for c in cols if c in date_opts), cols[0])
        f_desc = next((c for c in cols if c in desc_opts), cols[1])
        f_amt = next((c for c in cols if c in amt_opts), cols[2])

        df = df.rename(columns={f_date: 'Date', f_desc: 'Description', f_amt: 'Amount'})
        
        # Speciale ING Logica: Af/Bij afhandeling
        if 'Af Bij' in cols:
            df['Amount'] = df.apply(lambda x: -float(x['Amount']) if x['Af Bij'] == 'Af' else float(x['Amount']), axis=1)

    return df[['Date', 'Description', 'Amount']]

# --- 3. MACHINE LEARNING LOGICA ---
@st.cache_resource
def get_pipeline():
    """Bouwt de tekst-classificatie pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), # Zet tekst om in getallen
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42)) # Het brein
    ])

def train_model(data):
    """Traint het model op Description en Category."""
    model = get_pipeline()
    X = data['Description'].astype(str)
    y = data['Category'].astype(str)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

# --- 4. STREAMLIT FRONTEND ---
st.title("🤖 Slimme Grootboek Agent")
st.markdown("Categoriseer transacties automatisch naar grootboekrekeningen met Machine Learning.")

tab1, tab2 = st.tabs(["🧠 Model Trainen", "🚀 Voorspellingen & Dashboard"])

with tab1:
    st.header("Stap 1: De AI Leren")
    st.write("Upload je `boekhouding_training.csv` om de grootboekrekeningen te leren.")
    
    train_file = st.file_uploader("Upload Trainingsdata", type="csv", key="trainer")
    
    if train_file:
        df_train = pd.read_csv(train_file)
        if st.button("Start Leerproces"):
            with st.spinner("Model wordt getraind..."):
                train_model(df_train)
                st.success("Succes! De AI snapt nu je grootboekstructuur.")

with tab2:
    st.header("Stap 2: Automatisch Boekhouden")
    
    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ Geen getraind model gevonden. Ga naar de eerste tab.")
    else:
        new_file = st.file_uploader("Upload nieuwe bank-export (CSV)", type="csv", key="predictor")
        
        if new_file:
            # 1. Inladen en Standardiseren
            raw_new = pd.read_csv(new_file)
            mapped_df = standardize_df(raw_new)
            
            # 2. Voorspellen
            model = joblib.load(MODEL_FILE)
            predictions = model.predict(mapped_df['Description'].astype(str))
            mapped_df['Grootboekrekening'] = predictions
            
            st.success("Voorspelling voltooid!")
            st.dataframe(mapped_df, width="stretch") # 2026 syntax

            # --- 5. VISUALISATIE: DASHBOARD ---
            st.divider()
            st.subheader("📊 Uitgaven per Grootboekrekening")
            
            # Alleen kosten tonen (negatieve bedragen)
            expenses = mapped_df[mapped_df['Amount'] < 0].copy()
            expenses['Amount'] = expenses['Amount'].abs()
            
            if not expenses.empty:
                chart_data = expenses.groupby('Grootboekrekening')['Amount'].sum()
                st.bar_chart(chart_data)
            else:
                st.info("Geen uitgaven gevonden om te tonen in de grafiek.")
            
            # Download optie
            csv = mapped_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Boekhoudrapport", csv, "gecategoriseerd.csv")
