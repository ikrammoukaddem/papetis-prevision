"""
PAPETIS — Outil de prévision mensuelle des ventes
Auteure partie développement : Aya
CDC rédigé par : Ikram | Maquettes : Assma
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAPETIS — Outil de prévision",
    page_icon="📈",
    layout="centered",
)

# ── CSS thème sombre (fidèle aux maquettes d'Assma) ──────────────────────────
st.markdown("""
<style>
html, body, .stApp { background: #1a1f2e !important; color: #e2e8f0; }
.step-nav {
    display:flex; gap:8px; align-items:center;
    background:#242938; padding:10px 18px; border-radius:10px; margin-bottom:24px;
}
.step-pill { padding:5px 14px; border-radius:20px; font-size:13px; font-weight:600;
             color:#94a3b8; background:transparent; }
.step-pill.active { background:#3b82f6; color:#fff; }
.step-pill.done   { background:#2e3347; color:#94a3b8; }
.brand { font-weight:800; font-size:16px; margin-right:12px; }
.card  { background:#242938; border-radius:12px; padding:22px 24px; margin-bottom:16px; }
.dropzone { border:2px dashed #374151; border-radius:10px;
            padding:40px 24px; text-align:center; color:#94a3b8; margin-bottom:12px; }
.banner-ok   { background:#14532d; color:#86efac; border-radius:8px; padding:10px 16px; margin:8px 0; }
.banner-err  { background:#7f1d1d; color:#fca5a5; border-radius:8px; padding:10px 16px; margin:8px 0; }
.banner-warn { background:#78350f; color:#fef3c7; border-radius:8px; padding:10px 16px; margin:8px 0; }
.etape-label { font-size:12px; color:#60a5fa; font-weight:700; letter-spacing:1px; }
.etape-title { font-size:22px; font-weight:800; margin-top:4px; margin-bottom:18px; }
.stButton > button { background:#2e3347 !important; color:#e2e8f0 !important;
    border:1px solid #374151 !important; border-radius:8px !important;
    padding:8px 18px !important; font-size:14px !important; }
.stButton > button:hover { border-color:#3b82f6 !important; }
div[data-testid="stMetricValue"] { color:#60a5fa !important; font-size:20px !important; }
div[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:12px !important; }
h1,h2,h3,h4 { color:#e2e8f0 !important; }
p, li { color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "step":1, "df":None, "valid":False,
    "method":"MCO", "model":"Multiplicatif",
    "forecasts":None, "trend_eq":"",
    "coeff_aout":None, "coeff_mai":None,
    "outliers":[], "fig":None, "seasonal":None,
    "result":None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

MOIS_FR = ["Jan","Fév","Mar","Avr","Mai","Juin",
           "Juil","Aoû","Sep","Oct","Nov","Déc"]

# ── Barre de navigation ───────────────────────────────────────────────────────
def nav_bar():
    labels = ["1. Import","2. Paramètres","3. Résultats","4. Export"]
    pills = ""
    for i, l in enumerate(labels, 1):
        cls = "active" if i == st.session_state.step else "done"
        pills += f'<span class="step-pill {cls}">{l}</span>'
    st.markdown(
        f'<div class="step-nav"><span class="brand">△ PAPETIS — Outil de prévision</span>{pills}</div>',
        unsafe_allow_html=True)

# ── Chargement du fichier ─────────────────────────────────────────────────────
def load_file(uploaded):
    """
    Lit le vrai fichier PAPETIS (header sur ligne 3, colonnes Année/Mois/Ventes (kMAD)).
    Accepte aussi un CSV simple avec les mêmes colonnes.
    """
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            df.columns = [c.strip() for c in df.columns]
        else:
            # Essai 1 : feuille "Ventes globales" avec header=3 (format PAPETIS réel)
            try:
                df = pd.read_excel(uploaded, sheet_name="Ventes globales", header=3)
            except Exception:
                # Essai 2 : première feuille, header=0
                df = pd.read_excel(uploaded, sheet_name=0, header=0)
            df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        return None, f"Erreur de lecture : {e}"

    # Détecter les colonnes
    col_annee = next((c for c in df.columns if "ann" in c.lower()), None)
    col_mois  = next((c for c in df.columns if c.lower() == "mois"), None)
    col_vente = next((c for c in df.columns if "vente" in c.lower() or "kmad" in c.lower()), None)

    if not all([col_annee, col_mois, col_vente]):
        return None, "Colonnes non trouvées. Attendues : Année, Mois, Ventes (kMAD)."

    df = df[[col_annee, col_mois, col_vente]].copy()
    df.columns = ["annee", "mois", "ventes"]
    df = df.dropna(subset=["annee","mois","ventes"])
    df["annee"]  = pd.to_numeric(df["annee"],  errors="coerce")
    df["mois"]   = pd.to_numeric(df["mois"],   errors="coerce")
    df["ventes"] = pd.to_numeric(df["ventes"], errors="coerce")
    df = df.dropna().astype({"annee":int,"mois":int})
    df = df.sort_values(["annee","mois"]).reset_index(drop=True)

    if len(df) < 24:
        return None, f"Pas assez de données : {len(df)} lignes trouvées (minimum 24)."
    if df["ventes"].isna().any():
        return None, "Valeurs manquantes détectées dans la colonne Ventes."

    return df, None

# ── Calculs statistiques ──────────────────────────────────────────────────────
def compute_mco(series):
    n = len(series)
    t = np.arange(1, n+1)
    slope = (n*np.dot(t,series) - t.sum()*series.sum()) / (n*(t**2).sum() - t.sum()**2)
    intercept = (series.sum() - slope*t.sum()) / n
    return intercept, slope

def compute_mm12(series):
    return pd.Series(series).rolling(window=12, center=True).mean().values

def compute_seasonal(series, trend, model):
    n = len(series)
    monthly = np.full(12, np.nan)
    for m in range(12):
        vals = []
        for i in range(m, n, 12):
            if not np.isnan(trend[i]) and trend[i] != 0:
                v = series[i]/trend[i] if model=="Multiplicatif" else series[i]-trend[i]
                vals.append(v)
        if vals:
            monthly[m] = np.mean(vals)
    if model == "Multiplicatif":
        monthly = monthly / np.nanmean(monthly)
    else:
        monthly = monthly - np.nanmean(monthly)
    return monthly

def detect_outliers(series):
    mu, sigma = np.mean(series), np.std(series)
    return [(i, v, v-mu, v/mu) for i, v in enumerate(series) if abs(v-mu) > 2*sigma]

def run_forecast(df, method, model):
    series = df["ventes"].values
    n = len(series)

    # Tendance
    intercept, slope = compute_mco(series)
    trend_mco = intercept + slope * np.arange(1, n+1)

    if method == "MCO":
        trend = trend_mco
        eq = f"y = {intercept:.0f} + {slope:.0f}·t"
    else:
        trend_mm = compute_mm12(series)
        # Pour les NaN des bords, on complète avec MCO
        trend = np.where(np.isnan(trend_mm), trend_mco, trend_mm)
        eq = f"MM12  (tendance ≈ +{slope:.0f} kMAD/mois)"

    seasonal = compute_seasonal(series, trend, model)
    outliers  = detect_outliers(series)

    # Prévisions 12 mois
    forecasts = []
    last_annee = int(df["annee"].iloc[-1])
    last_mois  = int(df["mois"].iloc[-1])
    for k in range(1, 13):
        t_k      = n + k
        mois_k   = (last_mois + k - 1) % 12 + 1
        annee_k  = last_annee + (last_mois + k - 1) // 12
        trend_k  = intercept + slope * t_k
        coeff_k  = seasonal[(mois_k-1) % 12]
        val = trend_k * coeff_k if model=="Multiplicatif" else trend_k + coeff_k
        forecasts.append({
            "annee":annee_k, "mois":mois_k,
            "label": f"{MOIS_FR[mois_k-1]} {annee_k}",
            "tendance": trend_k, "coeff": coeff_k,
            "prevision": max(val, 0),
        })

    return {"trend":trend, "seasonal":seasonal, "eq":eq,
            "forecasts":forecasts, "outliers":outliers,
            "slope":slope, "intercept":intercept}

# ── Graphique ─────────────────────────────────────────────────────────────────
def make_figure(df, result):
    series = df["ventes"].values
    n = len(series)
    fc_vals = [f["prevision"] for f in result["forecasts"]]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#1a1f2e")
    ax.set_facecolor("#1a1f2e")

    ax.plot(range(n), series, color="#3b82f6", linewidth=2, label="Historique réel")
    ax.plot(range(n), result["trend"], color="#94a3b8", linewidth=1, linestyle="--", alpha=0.5)
    fc_x    = list(range(n-1, n+len(fc_vals)))
    fc_line = [series[-1]] + fc_vals
    ax.plot(fc_x, fc_line, color="#22d3ee", linewidth=2, linestyle="--", label="Prévision 2026")
    ax.axvline(x=n-1, color="#64748b", linestyle=":", linewidth=1.2)
    ax.text(n+0.3, max(series)*0.95, "→ 2026", color="#94a3b8", fontsize=9)

    for (idx, val, ecart, ratio) in result["outliers"]:
        ax.scatter(idx, val, color="#ef4444", s=90, zorder=5)
        lbl = f"{MOIS_FR[df['mois'].iloc[idx]-1]} {df['annee'].iloc[idx]}\n{val:.0f} kMAD"
        ax.annotate(lbl, xy=(idx,val), xytext=(idx+2, val+300),
                    fontsize=7.5, color="#f87171",
                    arrowprops=dict(arrowstyle="->", color="#f87171", lw=0.8))

    ax.spines[["top","right","left","bottom"]].set_color("#374151")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_ylabel("kMAD", color="#94a3b8", fontsize=9)
    ax.yaxis.grid(True, color="#2e3347", linewidth=0.7)
    ax.set_axisbelow(True)

    ticks = list(range(0, n, 6))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{MOIS_FR[df['mois'].iloc[t]-1]} {df['annee'].iloc[t]}" for t in ticks],
        fontsize=8, color="#94a3b8")

    handles = [
        mpatches.Patch(color="#3b82f6", label="Historique réel"),
        mpatches.Patch(color="#22d3ee", label="Prévision 2026"),
        mpatches.Patch(color="#ef4444", label="Observation atypique"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.2, labelcolor="white")
    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# ÉCRAN 1 — IMPORT
# ═══════════════════════════════════════════════════════════════════════════════
def screen_import():
    nav_bar()
    st.markdown('<div class="etape-label">ÉTAPE 1 / 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="etape-title">Importez votre fichier de ventes historiques</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="dropzone">📊<br><br>Glissez votre fichier Excel ou CSV ici</div></div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Parcourir…", type=["xlsx","csv"], label_visibility="collapsed")
    st.markdown('<p style="font-size:13px;">ℹ Formats acceptés : <b>.xlsx</b> et <b>.csv</b> — 60 observations attendues (Jan 2021 – Déc 2025)</p>',
                unsafe_allow_html=True)

    if uploaded:
        df, err = load_file(uploaded)
        if err:
            st.markdown(f'<div class="banner-err">⊘ <b>Format incorrect</b> — {err}</div>',
                        unsafe_allow_html=True)
            st.session_state.valid = False
        else:
            st.session_state.df    = df
            st.session_state.valid = True
            st.markdown(
                f'<div class="banner-ok">✓ <b>Fichier valide</b> — {len(df)} observations chargées, aucune valeur manquante détectée.</div>',
                unsafe_allow_html=True)
            st.dataframe(df.rename(columns={"annee":"Année","mois":"Mois","ventes":"Ventes (kMAD)"}).head(5),
                         use_container_width=True, hide_index=True)

    if st.session_state.valid:
        if st.button("→ Lancer l'analyse"):
            st.session_state.step = 2
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ÉCRAN 2 — PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════════
def screen_params():
    nav_bar()
    st.markdown('<div class="etape-label">ÉTAPE 2 / 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="etape-title">Configurez les paramètres de prévision</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Méthode de tendance**")
        method = st.radio("Méthode", ["Moindres carrés (MCO)","Moyennes mobiles (MM)"],
                          label_visibility="collapsed")
        st.caption("Les deux sont calculées et comparées")
    with col2:
        st.markdown("**Modèle de saisonnalité**")
        model = st.radio("Modèle", ["Multiplicatif ✅ Recommandé","Additif"],
                         label_visibility="collapsed")
        st.caption("Adapté à PAPETIS (pics proportionnels)")

    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("← Retour"):
            st.session_state.step = 1; st.rerun()
    with c2:
        if st.button("📊 Calculer les prévisions"):
            st.session_state.method = "MCO" if "MCO" in method else "MM"
            st.session_state.model  = "Multiplicatif" if "Multiplicatif" in model else "Additif"
            with st.spinner("Calcul en cours…"):
                result = run_forecast(st.session_state.df,
                                      st.session_state.method,
                                      st.session_state.model)
            st.session_state.result    = result
            st.session_state.forecasts = result["forecasts"]
            st.session_state.trend_eq  = result["eq"]
            st.session_state.seasonal  = result["seasonal"]
            st.session_state.outliers  = result["outliers"]
            st.session_state.coeff_aout = result["seasonal"][7]
            st.session_state.coeff_mai  = result["seasonal"][4]
            st.session_state.fig = make_figure(st.session_state.df, result)
            st.session_state.step = 3; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ÉCRAN 3 — RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════════════
def screen_results():
    nav_bar()
    st.markdown('<div class="etape-label">ÉTAPE 3 / 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="etape-title">Prévisions mensuelles — Janvier à Décembre 2026</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Tendance ({st.session_state.method})", st.session_state.trend_eq)
    c2.metric("Coeff. août (pic)",  f"× {st.session_state.coeff_aout:.2f}")
    c3.metric("Coeff. mai (creux)", f"× {st.session_state.coeff_mai:.2f}")

    if st.session_state.outliers:
        df = st.session_state.df
        lines = ""
        for (idx, val, ecart, ratio) in st.session_state.outliers:
            m = MOIS_FR[df["mois"].iloc[idx]-1]
            a = df["annee"].iloc[idx]
            lines += f"— {m} {a} : {val:.0f} kMAD (écart {ecart:+.0f} kMAD vs tendance, ratio × {ratio:.2f})<br>"
        st.markdown(
            f'<div class="banner-warn">⚠ <b>{len(st.session_state.outliers)} observations atypiques détectées</b><br>'
            f'{lines}Ces valeurs dépassent le seuil de 2 écarts-types. Elles sont conservées mais signalées.</div>',
            unsafe_allow_html=True)

    st.pyplot(st.session_state.fig, use_container_width=True)

    st.markdown("#### Tableau des prévisions 2026")
    fc_df = pd.DataFrame(st.session_state.forecasts)[["label","tendance","coeff","prevision"]]
    fc_df.columns = ["Mois","Tendance (kMAD)","Coeff. saisonnier","Prévision (kMAD)"]
    st.dataframe(fc_df.round(1), use_container_width=True, hide_index=True)

    st.markdown("#### Coefficients saisonniers")
    s = st.session_state.seasonal
    st.dataframe(
        pd.DataFrame({"Mois": MOIS_FR, "Coeff.": [round(x,3) for x in s]}).T,
        use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("← Retour"):
            st.session_state.step = 2; st.rerun()
    with c2:
        if st.button("→ Exporter les résultats"):
            st.session_state.step = 4; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ÉCRAN 4 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def generate_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    T1 = ParagraphStyle("T1", parent=styles["Heading1"], fontSize=16,
                         textColor=colors.HexColor("#1d4ed8"), spaceAfter=4)
    T2 = ParagraphStyle("T2", parent=styles["Heading2"], fontSize=12,
                         textColor=colors.HexColor("#1e40af"), spaceAfter=4)
    BD = styles["BodyText"]

    story = []
    story.append(Paragraph("PAPETIS DISTRIBUTION SARL", T1))
    story.append(Paragraph("Rapport de prévision mensuelle des ventes — 2026", T2))
    story.append(Paragraph(
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} | "
        f"Méthode : {st.session_state.method} | Modèle : {st.session_state.model}", BD))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("1. Tendance détectée", T2))
    story.append(Paragraph(f"Équation : <b>{st.session_state.trend_eq}</b>", BD))
    story.append(Paragraph(f"Coefficient août (pic rentrée) : <b>× {st.session_state.coeff_aout:.2f}</b>", BD))
    story.append(Paragraph(f"Coefficient mai (creux) : <b>× {st.session_state.coeff_mai:.2f}</b>", BD))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("2. Observations atypiques", T2))
    df = st.session_state.df
    for (idx, val, ecart, ratio) in st.session_state.outliers:
        m = MOIS_FR[df["mois"].iloc[idx]-1]
        a = df["annee"].iloc[idx]
        story.append(Paragraph(
            f"• {m} {a} : {val:.0f} kMAD — écart {ecart:+.0f} kMAD (ratio × {ratio:.2f}). Conservée, signalée.", BD))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("3. Graphique historique + prévisions 2026", T2))
    img_buf = io.BytesIO()
    st.session_state.fig.savefig(img_buf, format="png", dpi=150,
                                  bbox_inches="tight", facecolor="#1a1f2e")
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=16*cm, height=7*cm))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("4. Tableau des prévisions 2026", T2))
    header = ["Mois","Tendance (kMAD)","Coeff.","Prévision (kMAD)"]
    data = [header] + [
        [f["label"], f"{f['tendance']:.1f}", f"{f['coeff']:.3f}", f"{f['prevision']:.1f}"]
        for f in st.session_state.forecasts
    ]
    tbl = Table(data, colWidths=[4*cm,4*cm,3*cm,4*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("FONTSIZE",  (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#eff6ff")]),
        ("GRID",     (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
        ("ALIGN",    (1,1),(-1,-1), "CENTER"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("5. Recommandations opérationnelles", T2))
    for r in [
        f"Passer les commandes en mars–avril pour la rentrée août–septembre (pic × {st.session_state.coeff_aout:.1f}).",
        "Réduire les commandes pour mai–juin (creux saisonnier).",
        "Surveiller août 2026 : fort coefficient saisonnier — prévoir une marge de sécurité.",
        "Exporter ce tableau dans l'outil de gestion des achats pour piloter les réassorts.",
    ]:
        story.append(Paragraph(f"• {r}", BD))

    doc.build(story)
    buf.seek(0)
    return buf.read()

def screen_export():
    nav_bar()
    st.markdown('<div class="etape-label">ÉTAPE 4 / 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="etape-title">Exportez vos résultats de prévision</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card" style="text-align:center"><div style="font-size:40px">📗</div>'
                    '<h4>Fichier Excel</h4><p>Prévisions, tendance, coefficients, observations atypiques</p></div>',
                    unsafe_allow_html=True)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            fc_df = pd.DataFrame(st.session_state.forecasts)[["label","tendance","coeff","prevision"]]
            fc_df.columns = ["Mois","Tendance (kMAD)","Coeff. saisonnier","Prévision (kMAD)"]
            fc_df.round(1).to_excel(writer, sheet_name="Prévisions 2026", index=False)
            s = st.session_state.seasonal
            pd.DataFrame({"Mois":MOIS_FR,"Coeff. saisonnier":s}).to_excel(
                writer, sheet_name="Coefficients saisonniers", index=False)
            df = st.session_state.df
            if st.session_state.outliers:
                rows = [{"Mois":MOIS_FR[df["mois"].iloc[i]-1],"Année":df["annee"].iloc[i],
                         "Ventes (kMAD)":v,"Écart":e,"Ratio":r}
                        for i,v,e,r in st.session_state.outliers]
                pd.DataFrame(rows).to_excel(writer, sheet_name="Observations atypiques", index=False)
            df.rename(columns={"annee":"Année","mois":"Mois","ventes":"Ventes (kMAD)"}).to_excel(
                writer, sheet_name="Historique", index=False)
        st.download_button("⬇ Télécharger .xlsx", data=buf.getvalue(),
                           file_name="PAPETIS_previsions_2026.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    with col2:
        st.markdown('<div class="card" style="text-align:center"><div style="font-size:40px">📄</div>'
                    '<h4>Rapport PDF</h4><p>Rapport complet : graphique, analyse saisonnière, recommandations</p></div>',
                    unsafe_allow_html=True)
        st.download_button("⬇ Télécharger .pdf", data=generate_pdf(),
                           file_name="PAPETIS_rapport_previsions_2026.pdf",
                           mime="application/pdf",
                           use_container_width=True)

    st.markdown(
        f'<div class="banner-ok">✓ Analyse terminée — Prévisions Jan–Déc 2026 générées avec succès '
        f'(modèle {st.session_state.model}, {st.session_state.method})</div>',
        unsafe_allow_html=True)

    if st.button("← Retour"):
        st.session_state.step = 3; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
{1: screen_import, 2: screen_params, 3: screen_results, 4: screen_export}[st.session_state.step]()
