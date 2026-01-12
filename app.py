import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 1. CONFIGURATIE ---
st.set_page_config(page_title="Grootboek Agent Pro 2026", layout="wide", page_icon="🏦")
MODEL_FILE = 'trained_model.joblib'

# Uitgebreide lijst met grootboekrekeningen (RGS-gebaseerd)
GROOTBOEK_OPTIES = [
    "8000 Omzet (21% BTW)", "8100 Omzet (9% BTW)", "8200 Omzet (0% / Export)",
    "7000 Inkoopwaarde van de omzet", "7100 Verzendkosten inkoop",
    "4000 Huur kantoorruimte", "4010 Gas, water, licht", "4020 Schoonmaak & Onderhoud",
    "4100 Kantoorbenodigdheden", "4110 Software & SaaS abonnementen",
    "4120 Telefoon & Internet", "4130 Portokosten & Pakketten",
    "4200 Marketing & Advertenties", "4210 Representatiekosten & Relatiegeschenken", "4220 Website & Hosting",
    "4300 Brandstof & Laden", "4310 Onderhoud & Reparaties Auto", "4320 Parkeerkosten", "4330 Openbaar Vervoer & Taxi",
    "4400 Bankkosten & Transactiefees", "4410 Verzekeringen (Zakelijk)", "4420 Advieskosten (Boekhouder)",
    "4430 Kantinekosten & Lunches", "4440 Studiekosten & Training",
    "0000 Inventaris & Apparatuur", "1000 Bank / Kruisposten", "1400 BTW Afdracht / Ontvangst",
    "1600 Crediteuren", "2000 Privéstortingen", "2010 Privéopnamen"
]

# --- 2. ROBUUSTE KOLOM-MAPPER ---
def standardize_df(df):
    """Vertaalt diverse bank-exports naar een uniform formaat."""
    cols = df.columns.tolist()
    
    # Mapping logica voor diverse stijlen (ING, Rabo, Bunq, Fintech)
    date_opts = ['Datum', 'Date', 'Timestamp', 'Transactiedatum']
    desc_opts = ['Omschrijving', 'Description', 'Counterparty', 'Naam / Omschrijving', 'Naam tegenpartij', 'Reference']
    amt_opts = ['Bedrag', 'Amount', 'Amount_EUR', 'Bedrag (EUR)']

    f_date = next((c for c in cols if c in date_opts), None)
    # Voor omschrijving: check of zowel Counterparty als Reference er zijn (Fintech stijl)
    if 'Counterparty' in cols and 'Reference' in cols:
        df['Combined_Desc'] = df['Counterparty'].fillna('') + " " + df['Reference'].fillna('')
        f_desc = 'Combined_Desc'
    else:
        f_desc = next((c for c in cols if c in desc_opts), None)
    
    f_amt = next((c for c in cols if c in amt_opts), None)

    clean_df = pd.DataFrame()
    clean_df['Date'] = df[f_date] if f_date else "2026-01-12"
    clean_df['Description'] = df[f_desc].fillna("Onbekend") if f_desc else df.iloc[:, 0].fillna("Onbekend")
    
    if f_amt:
        clean_df['Amount'] = pd.to_numeric(df[f_amt], errors='coerce').fillna(0.0)
        # ING "Af Bij" logica
        if 'Af Bij' in cols:
            clean_df['Amount'] = df.apply(lambda x: -abs(x[f_amt]) if x['Af Bij'] == 'Af' else abs(x[f_amt]), axis=1)
    else:
        clean_df['Amount'] = 0.0

    if 'Category' in df.columns:
        clean_df['Category'] = df['Category']
    
    return clean_df

# --- 3. MACHINE LEARNING ENGINE ---
@st.cache_resource
def get_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

# --- 4. FRONTEND UI ---
st.title("🤖 Slimme Grootboek Agent Pro")
st.markdown("Automatiseer je boekhouding met AI-gebaseerde grootboek-voorspellingen.")

tab1, tab2 = st.tabs(["🧠 Leerproces & Correctie", "🚀 Voorspellen & Dashboard"])

# TAB 1: TRAINING & HUMAN-IN-THE-LOOP
with tab1:
    st.header("1. Train de AI")
    st.write("Upload een (deels) gecategoriseerde lijst om de AI jouw grootboekstructuur te leren.")
    
    train_file = st.file_uploader("Upload Trainingsdata (CSV)", type="csv", key="tr_up")

    if train_file:
        df_raw = pd.read_csv(train_file)
        df_to_review = standardize_df(df_raw)

        if 'Category' not in df_to_review.columns:
            df_to_review['Category'] = GROOTBOEK_OPTIES[8] # Standaard: Kantoorkosten

        st.subheader("📝 Review & Bulk-Correctie")
        search = st.text_input("🔍 Filter op omschrijving (bijv. 'Shell' of 'Salaris')")
        
        display_df = df_to_review
        if search:
            display_df = df_to_review[df_to_review['Description'].str.contains(search, case=False)]

        edited_df = st.data_editor(
            display_df,
            column_config={
                "Category": st.column_config.SelectboxColumn("Grootboekrekening", options=GROOTBOEK_OPTIES, required=True),
                "Amount": st.column_config.NumberColumn(format="€ %.2f")
            },
            hide_index=True,
            width="stretch"
        )

        if st.button("✅ Bevestig & Train AI"):
            with st.spinner("Model wordt getraind..."):
                # We trainen op de data zoals die nu in de editor staat
                X = edited_df['Description'].astype(str)
                y = edited_df['Category'].astype(str)
                
                model = get_pipeline()
                model.fit(X, y)
                joblib.dump(model, MODEL_FILE)
                st.success("Het model is succesvol getraind en klaar voor gebruik!")

# TAB 2: VOORSPELLING & ANALYSE
with tab2:
    st.header("2. Automatisch Categoriseren")
    
    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ Geen getraind model gevonden. Ga naar de eerste tab.")
    else:
        predict_file = st.file_uploader("Upload nieuwe banktransacties", type="csv", key="pr_up")
        
        if predict_file:
            df_new_raw = pd.read_csv(predict_file)
            df_mapped = standardize_df(df_new_raw)
            
            if st.button("🚀 Voorspel Grootboekrekeningen"):
                model = joblib.load(MODEL_FILE)
                df_mapped['Grootboek_Voorspelling'] = model.predict(df_mapped['Description'].astype(str))
                
                st.success("Analyse voltooid!")
                st.dataframe(df_mapped, width="stretch")

                # DASHBOARD
                st.divider()
                st.subheader("📊 Kostenanalyse")
                expenses = df_mapped[df_mapped['Amount'] < 0].copy()
                expenses['Amount'] = expenses['Amount'].abs()
                
                if not expenses.empty:
                    chart_data = expenses.groupby('Grootboek_Voorspelling')['Amount'].sum()
                    st.bar_chart(chart_data)
                
                csv = df_mapped.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Rapport", csv, "gecategoriseerd_rapport.csv")
