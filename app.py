import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 1. CONFIGURATIE & OPTIES ---
st.set_page_config(page_title="Grootboek Agent Pro 2026", layout="wide", page_icon="🏦")
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
    "4500 Reis- en verblijfkosten",
    "7000 Inkoop",
    "8000 Omzet"
]

# --- 2. DE ROBUUSTE MAPPER (VOORKOMT KEYERRORS) ---
def standardize_df(df):
    """
    Zet kolommen om naar Date, Description, Amount.
    Als kolommen ontbreken, worden ze veilig aangemaakt.
    """
    cols = df.columns.tolist()
    
    # Zoek naar Date, Description en Amount opties
    date_opts = ['Datum', 'Date', 'Timestamp', 'Transactiedatum']
    desc_opts = ['Omschrijving', 'Description', 'Counterparty', 'Naam / Omschrijving', 'Naam tegenpartij']
    amt_opts = ['Bedrag', 'Amount', 'Amount_EUR', 'Bedrag (EUR)']

    f_date = next((c for c in cols if c in date_opts), None)
    f_desc = next((c for c in cols if c in desc_opts), None)
    f_amt = next((c for c in cols if c in amt_opts), None)

    clean_df = pd.DataFrame()

    # Datum: Gebruik gevonden datum of vul dummy in
    if f_date:
        clean_df['Date'] = df[f_date]
    else:
        clean_df['Date'] = "2026-01-12"

    # Omschrijving: Zoek tekst of gebruik eerste kolom
    if f_desc:
        clean_df['Description'] = df[f_desc].fillna("Onbekend")
    else:
        clean_df['Description'] = df.iloc[:, 0].fillna("Onbekend")

    # Bedrag: Zoek bedrag of gebruik 0.0
    if f_amt:
        # Verwijder eventuele tekst uit bedragen en zet om naar float
        clean_df['Amount'] = pd.to_numeric(df[f_amt], errors='coerce').fillna(0.0)
    else:
        clean_df['Amount'] = 0.0

    # Speciale ING Af/Bij afhandeling indien aanwezig
    if 'Af Bij' in cols:
        clean_df['Amount'] = df.apply(
            lambda x: -abs(x[f_amt]) if x['Af Bij'] == 'Af' else abs(x[f_amt]), 
            axis=1
        )

    # Behoud Category als die al in de CSV staat (voor training)
    if 'Category' in df.columns:
        clean_df['Category'] = df['Category']
    
    return clean_df

# --- 3. MACHINE LEARNING ENGINE ---
@st.cache_resource
def get_pipeline():
    """Bouwt de tekst-classificatie pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

# --- 4. STREAMLIT FRONTEND ---
st.title("🤖 Slimme Grootboek Agent")
st.markdown("---")

tab1, tab2 = st.tabs(["🧠 Stap 1: Leerproces & Correctie", "🚀 Stap 2: Voorspellen & Rapportage"])

# --- TAB 1: TRAINING ---
with tab1:
    st.header("Data Review & AI Training")
    st.write("Upload je data, controleer de categorieën en train je model.")

    train_file = st.file_uploader("Upload Trainingsdata (CSV)", type="csv", key="train_upload")

    if train_file:
        df_raw = pd.read_csv(train_file)
        df_to_review = standardize_df(df_raw)

        # Zorg dat de kolom 'Category' altijd bestaat voor de editor
        if 'Category' not in df_to_review.columns:
            df_to_review['Category'] = GROOTBOEK_OPTIES[0]

        st.subheader("📝 Controleer en pas aan")
        
        # Zoekbalk filter
        search = st.text_input("🔍 Zoek in transacties (bijv. 'Albert' of 'Factuur')")
        display_df = df_to_review
        if search:
            display_df = df_to_review[df_to_review['Description'].str.contains(search, case=False)]

        # De Data Editor met Dropdown
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Grootboekrekening",
                    options=GROOTBOEK_OPTIES,
                    required=True,
                ),
                "Amount": st.column_config.NumberColumn(format="€ %.2f")
            },
            hide_index=True,
            width="stretch"
        )

        if st.button("✅ Bevestig & Train AI"):
            with st.spinner("De AI leert van jouw data..."):
                # We trainen op de GEHELE dataset die in de editor staat
                X = edited_df['Description'].astype(str)
                y = edited_df['Category'].astype(str)
                
                model = get_pipeline()
                model.fit(X, y)
                joblib.dump(model, MODEL_FILE)
                st.success("Het model is succesvol getraind en opgeslagen!")

# --- TAB 2: PREDICTION ---
with tab2:
    st.header("Automatische Voorspelling")
    
    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ Geen model gevonden. Train eerst de AI in de eerste tab.")
    else:
        predict_file = st.file_uploader("Upload nieuwe banktransacties", type="csv", key="predict_upload")
        
        if predict_file:
            df_new_raw = pd.read_csv(predict_file)
            df_mapped = standardize_df(df_new_raw)
            
            if st.button("🚀 Voorspel Grootboekrekeningen"):
                with st.spinner("AI analyseert transacties..."):
                    model = joblib.load(MODEL_FILE)
                    # Voorspelling op basis van Description
                    df_mapped['Voorspelling'] = model.predict(df_mapped['Description'].astype(str))
                    
                    st.success("Analyse voltooid!")
                    st.dataframe(df_mapped, width="stretch") # 2026 syntax
                    
                    # --- DASHBOARD ---
                    st.divider()
                    st.subheader("📊 Kosten per Grootboekrekening")
                    
                    # Alleen negatieve bedragen (kosten) tonen
                    expenses = df_mapped[df_mapped['Amount'] < 0].copy()
                    expenses['Amount'] = expenses['Amount'].abs()
                    
                    if not expenses.empty:
                        chart_data = expenses.groupby('Voorspelling')['Amount'].sum()
                        st.bar_chart(chart_data)
                    
                    # Download
                    csv = df_mapped.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Boekhoudrapport", csv, "gecategoriseerd_rapport.csv")
