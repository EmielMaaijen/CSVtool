import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- CONFIGURATIE ---
st.set_page_config(page_title="Slimme Bank Agent", layout="wide")
MODEL_FILE = 'trained_model.joblib'

# --- FUNCTIES ---

def train_and_save_model(data):
    """Traint een model op basis van handmatig gecategoriseerde data."""
    # We gebruiken 'Description' als input en 'Category' als doel
    X = data['Description'].astype(str)
    y = data['Category'].astype(str)
    
    # De Pipeline bundelt tekstverwerking en het algoritme
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), # Kijkt naar woorden en woordcombinaties
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE) # Sla het model op voor later gebruik
    return model

def load_model():
    """Laadt het model van de harde schijf als het bestaat."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

# --- FRONTEND UI ---

st.title("🔮 AI Transactie Voorspeller")
st.write("Stap 1: Leer de AI jouw gewoontes. Stap 2: Laat de AI nieuwe data categoriseren.")

tab1, tab2 = st.tabs(["🧠 Model Trainen", "🚀 Voorspellingen Doen"])

with tab1:
    st.header("Leerfase")
    st.write("Upload een CSV die je al een keer handmatig hebt gecategoriseerd.")
    
    train_file = st.file_uploader("Upload Trainingsdata (CSV met 'Description' en 'Category')", type="csv", key="train")
    
    if train_file:
        df_train = pd.read_csv(train_file)
        if st.button("Start Leerproces"):
            with st.spinner("De AI bestudeert je uitgaven..."):
                train_and_save_model(df_train)
                st.success("Klaar! De AI begrijpt nu hoe jij je geld uitgeeft.")

with tab2:
    st.header("Voorspelfase")
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Je moet eerst een model trainen in de eerste tab voordat je kunt voorspellen.")
    else:
        new_file = st.file_uploader("Upload Nieuwe Transacties (CSV)", type="csv", key="predict")
        
        if new_file:
            df_new = pd.read_csv(new_file)
            
            # Hier gebruiken we de 'Description' kolom van de nieuwe data
            if st.button("Voorspel Categorieën"):
                # De AI doet de voorspelling
                predictions = model.predict(df_new['Description'].astype(str))
                df_new['AI_Categorie'] = predictions
                
                st.success("Voorspelling voltooid!")
                st.dataframe(df_new, width="stretch") # 2026 syntax
                
                # Download knop voor het resultaat
                csv = df_new.to_csv(index=False).encode('utf-8')
                st.download_button("Download Gecategoriseerde CSV", csv, "voorspellingen.csv", "text/csv")
