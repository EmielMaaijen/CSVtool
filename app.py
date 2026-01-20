import streamlit as st
import pandas as pd
import numpy as np
import io
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from streamlit_gsheets import GSheetsConnection

# ============================================================
# 1) CONFIGURATIE
# ============================================================
st.set_page_config(
    page_title="Boekhoud Agent Pro",
    layout="wide",
    page_icon="🏦",
)

# JOUW TABNAAM (worksheet) IN GOOGLE SHEETS:
WORKSHEET_NAME = "Blad1"

# Vast schema van je Google Sheet:
REQUIRED_COLS = ["Date", "Description", "Amount", "Category"]

GROOTBOEK_OPTIES = [
    "8000 Omzet (21% BTW)", "8100 Omzet (9% BTW)", "8200 Omzet (0% / Export)", "8400 Overige opbrengsten",
    "7000 Inkoopwaarde van de omzet", "7100 Verzendkosten inkoop",
    "4000 Huur kantoorruimte", "4010 Gas, water, licht", "4020 Schoonmaak & Onderhoud",
    "4100 Kantoorbenodigdheden", "4110 Software & SaaS abonnementen", "4120 Telefoon & Internet", "4130 Portokosten & Pakketten",
    "4200 Marketing & Advertenties (Google/FB)", "4210 Representatiekosten & Relatiegeschenken", "4220 Website & Hosting",
    "4300 Brandstof & Laden", "4310 Onderhoud & Reparaties Auto", "4320 Parkeerkosten", "4330 Openbaar Vervoer & Taxi",
    "4400 Bankkosten & Transactiefees", "4410 Verzekeringen (Zakelijk)", "4420 Advieskosten (Boekhouder/Juridisch)",
    "4430 Kantinekosten & Lunches", "4440 Studiekosten & Training",

    # NIEUW: Personeelskosten
    "4500 Brutolonen / Salarissen", "4510 Sociale lasten (Loonheffing)", "4520 Pensioenpremies", "4530 Overige personeelskosten (WKR)",

    "0000 Inventaris & Apparatuur", "1000 Bank / Kruisposten", "1400 BTW Afdracht / Ontvangst",
    "1600 Crediteuren (Openstaande facturen)", "2000 Privéstortingen", "2010 Privéopnamen",
]

# ============================================================
# 2) GOOGLE SHEETS CONNECTIE
# ============================================================
@st.cache_resource
def get_conn():
    return st.connection("gsheets", type=GSheetsConnection)

def get_historical_data(conn: GSheetsConnection) -> pd.DataFrame:
    """
    Lees historische/gelabelde transacties uit Google Sheets (worksheet: WORKSHEET_NAME).
    """
    try:
        df = conn.read(worksheet=WORKSHEET_NAME, ttl="30s")
        if df is None or df.empty:
            st.warning(f"Worksheet '{WORKSHEET_NAME}' is leeg of niet gevonden.")
            return pd.DataFrame(columns=REQUIRED_COLS)
        return df
    except Exception as e:
        st.error(f"GSheets read faalt ({WORKSHEET_NAME}): {type(e).__name__}: {e}")
        return pd.DataFrame(columns=REQUIRED_COLS)

# ============================================================
# 3) ROBUUST CSV INLEZEN (UPLOADS)
# ============================================================
def read_csv_smart(uploaded_file) -> pd.DataFrame:
    """
    Robuuste CSV reader voor exports met:
    - delimiter detectie (, of ;)
    - encoding fallback (utf-8 -> latin1)
    - fallback naar python-engine
    - skip van corrupte regels i.p.v. crashen
    """
    raw = uploaded_file.getvalue()

    # Sniff delimiter op sample
    sample = raw[:4096].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        sep = dialect.delimiter
    except Exception:
        sep = ";" if sample.count(";") > sample.count(",") else ","

    # Encoding poging
    encodings = ["utf-8", "latin1"]

    last_err = None
    for enc in encodings:
        try:
            # 1) snelle C-engine
            return pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                engine="c",
                dtype=str,
                encoding=enc,
            )
        except Exception as e:
            last_err = e
            try:
                # 2) fallback python-engine + bad lines skip
                return pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep,
                    engine="python",
                    dtype=str,
                    encoding=enc,
                    on_bad_lines="skip",
                )
            except Exception as e2:
                last_err = e2

    raise last_err if last_err else ValueError("Onbekende CSV read fout.")

# ============================================================
# 4) DATA NORMALISATIE / SCHEMA
# ============================================================
def _parse_amount_series(series: pd.Series) -> pd.Series:
    """
    Accepteert o.a.:
    - 1234.56
    - 1.234,56
    - 1234,56
    """
    s = series.astype(str).fillna("").str.replace(" ", "", regex=False)

    # Als zowel '.' als ',' aanwezig: '.' is vermoedelijk duizendtalseparator
    mask = s.str.contains(",") & s.str.contains(r"\.", regex=True)
    s = s.where(~mask, s.str.replace(".", "", regex=False))

    # Zet decimal comma naar punt
    s = s.str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def ensure_schema_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dwing het historische Google Sheets schema af:
    Date, Description, Amount, Category
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]

    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = "" if c != "Amount" else 0.0

    out = out[REQUIRED_COLS].copy()
    out["Date"] = out["Date"].astype(str)
    out["Description"] = out["Description"].fillna("Onbekend").astype(str)
    out["Amount"] = _parse_amount_series(out["Amount"])
    out["Category"] = out["Category"].fillna("").astype(str)

    return out

def ensure_schema_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dwing een inference schema af (Category niet verplicht):
    Date, Description, Amount
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Description", "Amount"])

    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]

    for c in ["Date", "Description", "Amount"]:
        if c not in out.columns:
            out[c] = "" if c != "Amount" else 0.0

    out = out[["Date", "Description", "Amount"]].copy()
    out["Date"] = out["Date"].astype(str)
    out["Description"] = out["Description"].fillna("Onbekend").astype(str)
    out["Amount"] = _parse_amount_series(out["Amount"])

    return out

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliseer uploads (met variabele kolomnamen) naar:
    Date, Description, Amount, Category

    Belangrijk: map ook 'Grootboek' -> 'Category' zodat bestaande labels behouden blijven.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])

    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    cols = df.columns.tolist()

    date_opts = ["Datum", "Date", "Timestamp", "Transactiedatum"]
    desc_opts = ["Omschrijving", "Description", "Counterparty", "Naam / Omschrijving", "Reference"]
    amt_opts  = ["Bedrag", "Amount", "Amount_EUR", "Bedrag (EUR)"]

    # Cruciaal: Grootboek meenemen als labelbron
    cat_opts  = ["Category", "Grootboek", "Grootboekrekening", "Rekening", "Account", "Ledger"]

    f_date = next((c for c in cols if c in date_opts), None)
    f_desc = next((c for c in cols if c in desc_opts), None)
    f_amt  = next((c for c in cols if c in amt_opts), None)
    f_cat  = next((c for c in cols if c in cat_opts), None)

    clean = pd.DataFrame()
    clean["Date"] = df[f_date] if f_date else pd.Timestamp.today().date().isoformat()
    clean["Description"] = df[f_desc].fillna("Onbekend").astype(str) if f_desc else "Onbekend"
    clean["Amount"] = df[f_amt] if f_amt else 0.0

    if f_cat:
        clean["Category"] = df[f_cat]
    else:
        clean["Category"] = ""

    return clean

def df_fingerprint(df: pd.DataFrame) -> str:
    """
    Stabiele hash voor caching (retrain alleen bij wijziging sheets-data).
    """
    if df is None or df.empty:
        return "EMPTY"
    tmp = df[REQUIRED_COLS].copy()
    for c in tmp.columns:
        tmp[c] = tmp[c].astype(str)
    h = pd.util.hash_pandas_object(tmp, index=False).sum()
    return f"{len(tmp)}-{h}"

# ============================================================
# 5) MACHINE LEARNING (AUTO-TRAIN)
# ============================================================
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=250, random_state=42)),
    ])

@st.cache_resource
def train_model_cached(fingerprint: str, descriptions: tuple, categories: tuple):
    model = build_pipeline()
    model.fit(list(descriptions), list(categories))
    return model

def get_or_train_model(history_df: pd.DataFrame):
    """
    Train automatisch op basis van sheets-data.
    """
    hist = ensure_schema_history(history_df)

    # Alleen geldige labels gebruiken
    hist["Category"] = hist["Category"].fillna("").astype(str)
    hist = hist[hist["Description"].astype(str).str.len() > 0]
    hist = hist[hist["Category"].astype(str).str.strip().str.len() > 0]

    if len(hist) < 10:
        return None, "Te weinig gelabelde transacties (minimaal ~10) om stabiel te classificeren."

    fp = df_fingerprint(hist)
    descriptions = tuple(hist["Description"].astype(str).tolist())
    categories = tuple(hist["Category"].astype(str).tolist())

    model = train_model_cached(fp, descriptions, categories)
    return model, None

# ============================================================
# 6) UI
# ============================================================
st.title("🤖 Boekhoud Agent Pro")

page = st.sidebar.radio(
    "Navigatie",
    ["🚀 Analyse", "📊 Dashboard", "⚙️ Training & Beheer"],
    index=0,
)

with st.sidebar.expander("Onderhoud", expanded=False):
    if st.button("Clear cache / force refresh"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

conn = get_conn()
history_df = ensure_schema_history(get_historical_data(conn))

# ============================================================
# 7) PAGINA: ANALYSE (DEFAULT)
# ============================================================
if page == "🚀 Analyse":
    st.subheader("Analyse / Voorspellen (always-on)")

    colA, colB = st.columns([2, 1])
    with colA:
        st.caption("Upload een CSV om grootboek-categorieën te voorspellen op basis van historiek in Google Sheets.")
    with colB:
        st.info(f"AI-geheugen: {len(history_df)} transacties in Sheets ({WORKSHEET_NAME})")

    model, model_err = get_or_train_model(history_df)
    if model_err:
        st.warning(model_err)

    predict_file = st.file_uploader("Upload CSV voor analyse", type="csv", key="predict")

    with st.expander("Snelle check (1 transactie)", expanded=False):
        one_desc = st.text_input("Omschrijving", value="")
        one_amt = st.number_input("Bedrag (EUR)", value=0.0, step=1.0)
        if st.button("Voorspel voor 1 regel"):
            if model is None:
                st.error("Geen model beschikbaar (controleer Sheets-data / labels).")
            else:
                proba = model.predict_proba([str(one_desc)])[0]
                pred_idx = int(np.argmax(proba))
                pred = model.classes_[pred_idx]
                conf = float(proba[pred_idx])
                st.success(f"Voorspelling: {pred} (confidence: {conf:.2f})")

    if predict_file is not None:
        df_new_raw = read_csv_smart(predict_file)
        df_new_std = standardize_df(df_new_raw)
        df_new = ensure_schema_inference(df_new_std)

        st.write("Ingelezen bestand (gestandaardiseerd):")
        st.dataframe(df_new, use_container_width=True)

        if st.button("🚀 Start Analyse", type="primary"):
            if model is None:
                st.error("Analyse kan niet starten omdat er geen model beschikbaar is.")
            else:
                desc = df_new["Description"].astype(str).tolist()
                proba = model.predict_proba(desc)
                pred_idx = np.argmax(proba, axis=1)
                preds = model.classes_[pred_idx]
                confs = proba[np.arange(len(pred_idx)), pred_idx]

                df_out = df_new.copy()
                df_out["AI_Voorspelling"] = preds
                df_out["AI_Confidence"] = np.round(confs.astype(float), 3)

                st.success("Analyse afgerond.")
                st.dataframe(df_out, use_container_width=True)

                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download resultaat als CSV",
                    data=csv_bytes,
                    file_name="analyse_output.csv",
                    mime="text/csv",
                )

# ============================================================
# 8) PAGINA: DASHBOARD
# ============================================================
elif page == "📊 Dashboard":
    st.subheader("Dashboard (KPI’s)")

    if history_df is None or history_df.empty:
        st.warning("Geen data beschikbaar in Google Sheets.")
    else:
        hist = history_df.copy()
        hist["Date_parsed"] = pd.to_datetime(hist["Date"], errors="coerce")

        last_date = hist["Date_parsed"].max()
        last_date_str = last_date.date().isoformat() if pd.notna(last_date) else "Onbekend"

        total_tx = len(hist)
        total_abs = float(hist["Amount"].abs().sum())
        avg_amt = float(hist["Amount"].mean()) if total_tx > 0 else 0.0
        pos_tx = int((hist["Amount"] > 0).sum())
        neg_tx = int((hist["Amount"] < 0).sum())

        if pd.notna(last_date):
            cutoff = last_date - pd.Timedelta(days=30)
            last30 = hist[hist["Date_parsed"] >= cutoff]
            tx_30 = len(last30)
            vol_30 = float(last30["Amount"].abs().sum()) if len(last30) else 0.0
        else:
            tx_30, vol_30 = 0, 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transacties (totaal)", f"{total_tx}")
        k2.metric("Volume (abs)", f"€ {total_abs:,.2f}")
        k3.metric("Gem. bedrag", f"€ {avg_amt:,.2f}")
        k4.metric("Laatste transactiedatum", last_date_str)

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Inkomsten (+)", f"{pos_tx}")
        k6.metric("Uitgaven (-)", f"{neg_tx}")
        k7.metric("Transacties (30d)", f"{tx_30}")
        k8.metric("Volume (30d, abs)", f"€ {vol_30:,.2f}")

        st.markdown("### Categorieverdeling (Top 20)")
        cat_counts = hist["Category"].fillna("Onbekend").value_counts().head(20)
        st.bar_chart(cat_counts)

        st.markdown("### Top omschrijvingen (Top 15)")
        top_desc = hist["Description"].fillna("Onbekend").value_counts().head(15).reset_index()
        top_desc.columns = ["Description", "Count"]
        st.dataframe(top_desc, use_container_width=True)

# ============================================================
# 9) PAGINA: TRAINING & BEHEER (WEGGESTOPT)
# ============================================================
else:
    st.subheader("Training & Beheer (optioneel)")
    st.caption("Gebruik dit alleen om gelabelde data toe te voegen/te corrigeren. Analyse traint automatisch met Sheets.")

    st.info(f"AI-geheugen: {len(history_df)} transacties in Sheets ({WORKSHEET_NAME})")

    with st.expander("➕ Upload en label transacties (voor modelverbetering)", expanded=False):
        train_file = st.file_uploader("Upload CSV met nieuwe transacties", type="csv", key="train")

        if train_file is not None:
            raw = read_csv_smart(train_file)
            df_to_review = standardize_df(raw)

            # Zorg dat schema klopt
            df_to_review = ensure_schema_history(df_to_review)

            # Default alleen waar Category echt leeg is (niet alles overschrijven)
            df_to_review["Category"] = df_to_review["Category"].replace({None: "", np.nan: ""}).astype(str)
            empty_mask = df_to_review["Category"].str.strip().eq("")
            df_to_review.loc[empty_mask, "Category"] = GROOTBOEK_OPTIES[0]

            edited_df = st.data_editor(
                df_to_review,
                column_config={
                    "Category": st.column_config.SelectboxColumn(
                        "Grootboek",
                        options=GROOTBOEK_OPTIES,
                    )
                },
                hide_index=True,
                use_container_width=True,
            )

            if st.button("💾 Opslaan naar Sheets", type="primary"):
                updated = pd.concat([history_df, edited_df], ignore_index=True)
                updated = ensure_schema_history(updated)
                updated = updated.drop_duplicates(subset=["Date", "Description", "Amount", "Category"])

                conn.update(worksheet=WORKSHEET_NAME, data=updated)
                st.success("Opgeslagen. Analyse traint automatisch met de nieuwe data.")
