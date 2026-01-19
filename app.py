import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- FAILSAFE IMPORT VOOR GSHEETS ---
# Dit probeert beide namen van de module om importfouten te voorkomen
try:
    from streamlit_gsheets_connection import GSheetsConnection
except ImportError:
    try:
        from st_gsheets_connection import GSheetsConnection
    except ImportError:
        st.error("Fout: GSheets bibliotheek niet gevonden. Controleer requirements.txt.")

# --- 1. CONFIGURATIE ---
st.set_page_config(page_title="Zelflerende Boekhoud Agent 2026", layout="wide", page_icon="🏦")
MODEL_FILE = 'trained_model.joblib'

# DE VOLLEDIGE LIJST MET GROOTBOEKREKENINGEN
GROOTBOEK_OPTIES = [
    "8000 Omzet (21% BTW)", "8100 Omzet (9% BTW)", "8200 Omzet (0% / Export)", "8400 Overige opbrengsten",
    "7000 Inkoopwaarde van de omzet", "7100 Verzendkosten inkoop",
    "4000 Huur kantoorruimte", "4010 Gas, water, licht", "4020 Schoonmaak & Onderhoud",
    "4100 Kantoorbenodigdheden", "4110 Software & SaaS abonnementen", "4120 Telefoon & Internet", "4130 Portokosten & Pakketten",
    "4200 Marketing & Advertenties (Google/FB)", "4210 Representatiekosten & Relatiegeschenken", "4220 Website & Hosting",
    "4300 Brandstof & Laden", "4310 Onderhoud & Reparaties Auto", "4320 Parkeerkosten", "4330 Openbaar Vervoer & Taxi",
    "4400 Bankkosten & Transactiefees", "4410 Verzekeringen (Zakelijk)", "4420 Advieskosten (Boekhouder/Juridisch)",
    "4430 Kantinekosten & Lunches", "4440 Studiekosten & Training",
    "0000 Inventaris & Apparatuur", "1000 Bank / Kruisposten", "1400 BTW Afdracht / Ontvangst",
    "1600 Crediteuren (Openstaande facturen)", "2000 Privéstortingen", "2010 Privéopnamen"
]

# --- 2. GOOGLE SHEETS CONNECTIE ---
# Gebruik de directe Class-import voor maximale stabiliteit
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"Verbindingsfout met Google Sheets: {e}")

def get_historical_data():
    """Haalt alle eerder opgeslagen transacties op uit Google Sheets."""
    try:
        df = conn.read(ttl="1m")
        if df is not None and not df.empty:
            return df
        return pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])
    except Exception:
        return pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])

# --- 3. DATA NORMALISATIE ---
def standardize_df(df):
    cols = df.columns.tolist()
    date_opts = ['Datum', 'Date', 'Timestamp', 'Transactiedatum']
    desc_opts = ['Omschrijving', 'Description', 'Counterparty', 'Naam / Omschrijving', 'Naam tegenpartij', 'Reference']
    amt_opts = ['Bedrag', 'Amount', 'Amount_EUR', 'Bedrag (EUR)']

    f_date = next((c for c in cols if c in date_opts), None)
    if 'Counterparty' in cols and 'Reference' in cols:
        df['Combined_Desc'] = df['Counterparty'].fillna('') + " " + df['Reference'].fillna('')
        f_desc = 'Combined_Desc'
    else:
        f_desc = next((c for c in cols if c in desc_opts), None)
    
    f_amt = next((c for c in cols if c in amt_opts), None)

    clean_df = pd.DataFrame()
    clean_df['Date'] = df[f_date] if f_date else "2026-01-19"
    clean_df['Description'] = df[f_desc].fillna("Onbekend") if f_desc else df.iloc[:, 0].fillna("Onbekend")
    
    if f_amt:
        clean_df['Amount'] = pd.to_numeric(df[f_amt], errors='coerce').fillna(0.0)
    else:
        clean_df['Amount'] = 0.0
    
    if 'Category' in df.columns:
        clean_df['Category'] = df['Category']
    
    return clean_df

# --- 4. MACHINE LEARNING ENGINE ---
@st.cache_resource
def get_pipeline():
    """Bouwt de AI-pipeline voor tekstclassificatie."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

# --- 5. FRONTEND UI ---
st.title("🤖 Slimme Grootboek Agent Pro")
st.markdown("---")

tab1, tab2 = st.tabs(["🧠 Training & Geheugen", "🚀 Voorspellingen & Dashboard"])

with tab1:
    st.header("Stap 1: Review & Leerproces")
    history_df = get_historical_data()
    st.info(f"Aantal transacties in AI-geheugen: **{len(history_df)}**")

    train_file = st.file_uploader("Upload nieuwe data om te leren (CSV)", type="csv", key="train_up")

    if train_file:
        raw_data = pd.read_csv(train_file)
        df_to_review = standardize_df(raw_data)

        if 'Category' not in df_to_review.columns:
            df_to_review['Category'] = GROOTBOEK_OPTIES[0]

        st.subheader("📝 Controleer en pas aan")
        edited_df = st.data_editor(
            df_to_review,
            column_config={
                "Category": st.column_config.SelectboxColumn("Grootboekrekening", options=GROOTBOEK_OPTIES, required=True),
                "Amount": st.column_config.NumberColumn(format="€ %.2f")
            },
            hide_index=True,
            width="stretch"
        )

        if st.button("💾 Opslaan & AI Trainen"):
            with st.spinner("Data wordt opgeslagen..."):
                updated_history = pd.concat([history_df, edited_df], ignore_index=True).drop_duplicates()
                conn.update(data=updated_history)
                
                X = updated_history['Description'].astype(str)
                y = updated_history['Category'].astype(str)
                model = get_pipeline()
                model.fit(X, y)
                joblib.dump(model, MODEL_FILE)
                st.success("AI is weer een stukje slimmer!")

with tab2:
    st.header("Stap 2: Voorspellen")
    if not os.path.exists(MODEL_FILE) and history_df.empty:
        st.warning("⚠️ Geen kennis gevonden. Voeg eerst data toe in Tab 1.")
    else:
        if not os.path.exists(MODEL_FILE) and not history_df.empty:
            model = get_pipeline()
            model.fit(history_df['Description'].astype(str), history_df['Category'].astype(str))
            joblib.dump(model, MODEL_FILE)

        predict_file = st.file_uploader("Upload bankbestand voor analyse", type="csv", key="pred_up")
        if predict_file:
            df_new_raw = pd.read_csv(predict_file)
            df_mapped = standardize_df(df_new_raw)
            
            if st.button("🚀 Voorspel Grootboekrekeningen"):
                model = joblib.load(MODEL_FILE)
                df_mapped['AI_Voorspelling'] = model.predict(df_mapped['Description'].astype(str))
                st.success("Analyse voltooid!")
                st.dataframe(df_mapped, width="stretch")
