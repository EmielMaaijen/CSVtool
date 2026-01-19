import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 1. CONFIGURATIE ---
st.set_page_config(page_title="Zelflerende Boekhoud Agent Pro", layout="wide", page_icon="🏦")
MODEL_FILE = 'trained_model.joblib'

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
# We gebruiken de string "gsheets" zodat Streamlit zelf de bibliotheek zoekt
try:
    conn = st.connection("gsheets", type="gsheets")
except Exception as e:
    st.error(f"Kan geen verbinding maken: {e}")
    conn = None

def get_historical_data():
    if conn is None: return pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])
    try:
        df = conn.read(ttl="1m")
        return df if df is not None and not df.empty else pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])
    except:
        return pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])

# --- 3. DATA NORMALISATIE ---
def standardize_df(df):
    cols = df.columns.tolist()
    date_opts = ['Datum', 'Date', 'Timestamp', 'Transactiedatum']
    desc_opts = ['Omschrijving', 'Description', 'Counterparty', 'Naam / Omschrijving', 'Reference']
    amt_opts = ['Bedrag', 'Amount', 'Amount_EUR', 'Bedrag (EUR)']
    f_date = next((c for c in cols if c in date_opts), None)
    f_desc = next((c for c in cols if c in desc_opts), None)
    f_amt = next((c for c in cols if c in amt_opts), None)
    clean_df = pd.DataFrame()
    clean_df['Date'] = df[f_date] if f_date else "2026-01-19"
    clean_df['Description'] = df[f_desc].fillna("Onbekend") if f_desc else "Onbekend"
    clean_df['Amount'] = pd.to_numeric(df[f_amt], errors='coerce').fillna(0.0) if f_amt else 0.0
    if 'Category' in df.columns: clean_df['Category'] = df['Category']
    return clean_df

# --- 4. MACHINE LEARNING ---
@st.cache_resource
def get_pipeline():
    return Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('clf', RandomForestClassifier(n_estimators=100))])

# --- 5. UI ---
st.title("🤖 Boekhoud Agent Pro")
tab1, tab2 = st.tabs(["🧠 Training", "🚀 Voorspellen"])

with tab1:
    history_df = get_historical_data()
    st.info(f"AI-geheugen: {len(history_df)} transacties")
    train_file = st.file_uploader("Upload CSV", type="csv", key="tr")
    if train_file:
        df_to_review = standardize_df(pd.read_csv(train_file))
        if 'Category' not in df_to_review.columns: df_to_review['Category'] = GROOTBOEK_OPTIES[0]
        edited_df = st.data_editor(df_to_review, column_config={"Category": st.column_config.SelectboxColumn("Grootboek", options=GROOTBOEK_OPTIES)}, hide_index=True)
        if st.button("💾 Opslaan & Trainen"):
            updated = pd.concat([history_df, edited_df], ignore_index=True).drop_duplicates()
            conn.update(data=updated)
            model = get_pipeline()
            model.fit(updated['Description'].astype(str), updated['Category'].astype(str))
            joblib.dump(model, MODEL_FILE)
            st.success("Klaar!")

with tab2:
    if os.path.exists(MODEL_FILE):
        predict_file = st.file_uploader("Analyseer bestand", type="csv", key="pr")
        if predict_file:
            df_new = standardize_df(pd.read_csv(predict_file))
            if st.button("🚀 Start Analyse"):
                model = joblib.load(MODEL_FILE)
                df_new['AI_Voorspelling'] = model.predict(df_new['Description'].astype(str))
                st.dataframe(df_new)
