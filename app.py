"""
PAPETIS — Outil de prévision mensuelle des ventes
7 écrans — fidèle aux maquettes d'Assma (version finale)
Auteure : Aya
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

st.set_page_config(page_title="PAPETIS — Prévision", page_icon="📊", layout="centered")

# ══════════════════════════════════════════════════════════════════════
# CSS — thème sombre fidèle aux maquettes
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
html,body,.stApp{background:#0f1623 !important;color:#e2e8f0;}
.nav{display:flex;gap:6px;align-items:center;background:#1a2235;
     padding:9px 16px;border-radius:10px;margin-bottom:22px;flex-wrap:wrap;}
.np{padding:4px 12px;border-radius:16px;font-size:12px;font-weight:700;
    color:#64748b;background:transparent;}
.np.active{background:#3b82f6;color:#fff;}
.np.done{background:#1e2d45;color:#64748b;}
.brand{font-weight:900;font-size:14px;margin-right:10px;color:#e2e8f0;}
.el{font-size:11px;color:#60a5fa;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;}
.et{font-size:20px;font-weight:800;margin:4px 0 18px 0;}
.card{background:#1a2235;border-radius:10px;padding:20px;margin-bottom:14px;}
.card2{background:#111827;border-radius:10px;padding:20px;margin-bottom:14px;border:1px solid #1e2d45;}
.dz{border:2px dashed #2d3f55;border-radius:10px;padding:36px;text-align:center;color:#475569;}
.ok{background:#14532d;color:#86efac;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:13px;}
.er{background:#7f1d1d;color:#fca5a5;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:13px;}
.wa{background:#78350f;color:#fef3c7;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:13px;}
.info{background:#1e3a5f;color:#93c5fd;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:13px;}
.kpi{background:#1a2235;border-radius:10px;padding:14px 18px;text-align:center;}
.kpi-v{font-size:22px;font-weight:800;color:#60a5fa;}
.kpi-l{font-size:11px;color:#64748b;margin-top:4px;}
.badge-g{background:#14532d;color:#86efac;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;margin-left:6px;}
.badge-r{background:#7f1d1d;color:#fca5a5;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;margin-left:6px;}
.badge-b{background:#1e3a5f;color:#93c5fd;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;margin-left:6px;}
.sc-card{border-left:4px solid #3b82f6;background:#1a2235;border-radius:0 10px 10px 0;
         padding:16px;margin-bottom:12px;}
.sc-title{font-size:13px;font-weight:700;color:#60a5fa;}
.sc-val{font-size:26px;font-weight:900;color:#60a5fa;margin:6px 0;}
.sc-desc{font-size:12px;color:#64748b;}
.bar-wrap{display:flex;align-items:center;gap:10px;margin:6px 0;}
.bar-label{font-size:12px;color:#94a3b8;width:140px;flex-shrink:0;}
.bar-outer{flex:1;background:#1e2d45;border-radius:4px;height:12px;}
.bar-inner{height:12px;border-radius:4px;}
.bar-val{font-size:12px;color:#e2e8f0;width:90px;text-align:right;flex-shrink:0;}
.stButton>button{background:#1e2d45 !important;color:#e2e8f0 !important;
    border:1px solid #2d3f55 !important;border-radius:8px !important;
    padding:7px 16px !important;font-size:13px !important;}
.stButton>button:hover{border-color:#3b82f6 !important;}
div[data-testid="stMetricValue"]{color:#60a5fa !important;font-size:20px !important;}
div[data-testid="stMetricLabel"]{color:#64748b !important;font-size:11px !important;}
h1,h2,h3,h4{color:#e2e8f0 !important;}
p,li{color:#94a3b8;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "step":1,"df":None,"df_fam":None,"valid":False,
    "method":"MCO","model":"Multiplicatif",
    "result_mco":None,"result_mm":None,"result":None,
    "fam_results":None,
    "seasonal":None,"outliers":[],"trend_eq":"",
    "coeff_aout":None,"coeff_mai":None,
    "forecasts":None,"fig_results":None,
    "fig_comp":None,"fig_fam":None,
    "cv":None,
}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

MOIS_FR = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Aoû","Sep","Oct","Nov","Déc"]
FAM_COLORS = ["#3b82f6","#22c55e","#f59e0b","#a78bfa","#ef4444","#06b6d4"]
FAM_LABELS = {"Fort pic":"#ef4444","Pic":"#f59e0b","Modéré":"#a78bfa","Stable":"#22c55e"}

# ══════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════
STEPS = ["1. Import","2. Paramètres","3. Comparaison",
         "4. Résultats","5. Scénarios","6. Familles","7. Export"]

def nav_bar():
    pills = "".join(
        f'<span class="np {"active" if i+1==st.session_state.step else "done"}">{l}</span>'
        for i,l in enumerate(STEPS))
    st.markdown(
        f'<div class="nav"><span class="brand">📊 PAPETIS — Prévision</span>{pills}</div>',
        unsafe_allow_html=True)

def go(n): st.session_state.step=n; st.rerun()

# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ══════════════════════════════════════════════════════════════════════
def load_file(uploaded):
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            df.columns=[c.strip() for c in df.columns]
            df_fam=None
        else:
            xl=pd.ExcelFile(uploaded)
            df=pd.read_excel(xl,sheet_name="Ventes globales",header=3)
            df.columns=[str(c).strip() for c in df.columns]
            try:
                df_fam=pd.read_excel(xl,sheet_name="Ventes par famille",header=3)
                df_fam.columns=[str(c).strip() for c in df_fam.columns]
            except: df_fam=None
    except Exception as e: return None,None,str(e)

    ca=next((c for c in df.columns if "ann" in c.lower()),None)
    cm=next((c for c in df.columns if c.lower()=="mois"),None)
    cv=next((c for c in df.columns if "vente" in c.lower() or "kmad" in c.lower()),None)
    if not all([ca,cm,cv]):
        return None,None,"Colonnes non trouvées — attendues : Année, Mois, Ventes (kMAD)."

    df=df[[ca,cm,cv]].copy(); df.columns=["annee","mois","ventes"]
    df=df.dropna()
    df["annee"]=pd.to_numeric(df["annee"],errors="coerce")
    df["mois"]=pd.to_numeric(df["mois"],errors="coerce")
    df["ventes"]=pd.to_numeric(df["ventes"],errors="coerce")
    df=df.dropna().astype({"annee":int,"mois":int})
    df=df.sort_values(["annee","mois"]).reset_index(drop=True)
    if len(df)<24: return None,None,f"Pas assez de données : {len(df)} lignes."
    return df,df_fam,None

# ══════════════════════════════════════════════════════════════════════
# STATISTIQUES
# ══════════════════════════════════════════════════════════════════════
def mco(series):
    n=len(series); t=np.arange(1,n+1)
    s=(n*np.dot(t,series)-t.sum()*series.sum())/(n*(t**2).sum()-t.sum()**2)
    i=(series.sum()-s*t.sum())/n
    return i,s

def mm12(series):
    return pd.Series(series).rolling(12,center=True).mean().values

def seasonal_coeffs(series,trend,model):
    n=len(series); monthly=np.full(12,np.nan)
    for m in range(12):
        vals=[series[i]/trend[i] if model=="Multiplicatif" else series[i]-trend[i]
              for i in range(m,n,12) if not np.isnan(trend[i]) and trend[i]!=0]
        if vals: monthly[m]=np.mean(vals)
    if model=="Multiplicatif": monthly/=np.nanmean(monthly)
    else: monthly-=np.nanmean(monthly)
    return monthly

def errors(series,fitted):
    r=series-fitted
    mae=np.mean(np.abs(r))
    rmse=np.sqrt(np.mean(r**2))
    mape=np.mean(np.abs(r/series))*100
    return mae,rmse,mape

def outliers(series):
    mu,sig=np.mean(series),np.std(series)
    return [(i,v,v-mu,v/mu) for i,v in enumerate(series) if abs(v-mu)>2*sig]

def cv_test(series):
    """Coefficient de variation pour choisir modèle multiplicatif vs additif."""
    monthly_std=[]
    for m in range(12):
        vals=[series[i] for i in range(m,len(series),12)]
        if len(vals)>1: monthly_std.append(np.std(vals)/np.mean(vals))
    return np.mean(monthly_std)

def forecast(df,method,model,pct=0.0):
    series=df["ventes"].values; n=len(series)
    ic,sl=mco(series)
    trend_mco=ic+sl*np.arange(1,n+1)
    if method=="MCO":
        trend=trend_mco; eq=f"y = {ic:.0f} + {sl:.0f}·t"
    else:
        tm=mm12(series)
        trend=np.where(np.isnan(tm),trend_mco,tm)
        eq=f"MM12 (tendance ≈ +{sl:.0f} kMAD/mois)"

    seas=seasonal_coeffs(series,trend,model)
    fitted=np.array([trend[i]*seas[i%12] if model=="Multiplicatif"
                     else trend[i]+seas[i%12] for i in range(n)])
    mae,rmse,mape=errors(series,fitted)
    outs=outliers(series)
    std_res=np.std(series-fitted)

    last_a=int(df["annee"].iloc[-1]); last_m=int(df["mois"].iloc[-1])
    fcs=[]
    for k in range(1,13):
        tk=n+k; mk=(last_m+k-1)%12+1; ak=last_a+(last_m+k-1)//12
        tr=ic+sl*tk; co=seas[(mk-1)%12]
        v=tr*co if model=="Multiplicatif" else tr+co
        v=max(v,0); vs=v*(1+pct/100)
        fcs.append({"annee":ak,"mois":mk,"label":f"{MOIS_FR[mk-1]} {ak}",
                    "tendance":tr,"coeff":co,"prevision":v,
                    "prevision_sc":vs,
                    "ic_bas":max(vs-1.96*std_res,0),
                    "ic_haut":vs+1.96*std_res})
    return {"trend":trend,"seas":seas,"eq":eq,"fcs":fcs,"outs":outs,
            "mae":mae,"rmse":rmse,"mape":mape,"fitted":fitted,
            "slope":sl,"intercept":ic}

def forecast_famille(df_fam,method,model,pct=0.0):
    results={}
    ca=next((c for c in df_fam.columns if "ann" in c.lower()),None)
    cm=next((c for c in df_fam.columns if c.lower()=="mois"),None)
    skip={"annee","mois","nom du mois","t","total","unnamed"}
    fams=[c for c in df_fam.columns
          if c not in [ca,cm] and c.lower() not in skip
          and not c.lower().startswith("unnamed")]
    for fam in fams:
        try:
            sub=df_fam[[ca,cm,fam]].copy(); sub.columns=["annee","mois","ventes"]
            sub=sub.dropna()
            sub["annee"]=pd.to_numeric(sub["annee"],errors="coerce")
            sub["mois"]=pd.to_numeric(sub["mois"],errors="coerce")
            sub["ventes"]=pd.to_numeric(sub["ventes"],errors="coerce")
            sub=sub.dropna().astype({"annee":int,"mois":int})
            sub=sub.sort_values(["annee","mois"]).reset_index(drop=True)
            if len(sub)>=24: results[fam]=forecast(sub,method,model,pct)
        except: continue
    return results

# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════
BG="#0f1623"; BG2="#1a2235"; GRID="#1e2d45"

def fig_comparaison(df,r_mco,r_mm):
    series=df["ventes"].values; n=len(series)
    fig,ax=plt.subplots(figsize=(11,4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
    ax.plot(range(n),series,color="#94a3b8",lw=1.5,label="Réel")
    ax.plot(range(n),r_mco["fitted"],color="#3b82f6",lw=2,label=f"MCO (MAPE {r_mco['mape']:.1f}%)")
    ax.plot(range(n),r_mm["fitted"],color="#f59e0b",lw=2,linestyle="--",label=f"MM12 (MAPE {r_mm['mape']:.1f}%)")
    ticks=list(range(0,n,6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{MOIS_FR[df['mois'].iloc[t]-1]} {df['annee'].iloc[t]}"
                        for t in ticks],fontsize=8,color="#94a3b8")
    ax.spines[["top","right","left","bottom"]].set_color(GRID)
    ax.tick_params(colors="#94a3b8",labelsize=8)
    ax.yaxis.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
    ax.set_ylabel("kMAD",color="#94a3b8",fontsize=9)
    ax.legend(loc="upper left",fontsize=8,framealpha=0.15,labelcolor="white")
    fig.tight_layout(); return fig

def fig_results(df,result):
    series=df["ventes"].values; n=len(series)
    fcs=result["fcs"]
    fc_v=[f["prevision_sc"] for f in fcs]
    ic_b=[f["ic_bas"] for f in fcs]
    ic_h=[f["ic_haut"] for f in fcs]
    fig,ax=plt.subplots(figsize=(11,4.8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
    ax.plot(range(n),series,color="#e2e8f0",lw=2,label="Historique")
    fc_x=list(range(n-1,n+len(fc_v)))
    ax.fill_between(fc_x,[series[-1]]+ic_b,[series[-1]]+ic_h,
                    color="#22d3ee",alpha=0.12,label="IC 95%")
    ax.plot(fc_x,[series[-1]]+fc_v,color="#22d3ee",lw=2,
            linestyle="--",label="Prévision 2026")
    ax.axvline(x=n-1,color="#475569",linestyle=":",lw=1.2)
    ymax=max(max(series),max(fc_v))*1.05
    ax.text(n+0.2,ymax*0.92,"→ 2026",color="#94a3b8",fontsize=9)
    for (idx,val,ec,ra) in result["outs"]:
        ax.scatter(idx,val,color="#ef4444",s=100,zorder=5)
        ml=MOIS_FR[df["mois"].iloc[idx]-1]; an=df["annee"].iloc[idx]
        ax.annotate(f"{ml} {an}\n{val:.0f} kMAD",xy=(idx,val),
                    xytext=(idx+1.5,val+300),fontsize=7.5,color="#f87171",
                    arrowprops=dict(arrowstyle="->",color="#f87171",lw=0.8))
    ticks=list(range(0,n,6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{MOIS_FR[df['mois'].iloc[t]-1]} {df['annee'].iloc[t]}"
                        for t in ticks],fontsize=8,color="#94a3b8")
    ax.set_ylim(0,ymax)
    ax.spines[["top","right","left","bottom"]].set_color(GRID)
    ax.tick_params(colors="#94a3b8",labelsize=8)
    ax.yaxis.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
    ax.set_ylabel("kMAD",color="#94a3b8",fontsize=9)
    handles=[mpatches.Patch(color="#e2e8f0",label="Historique réel"),
             mpatches.Patch(color="#22d3ee",label="Prévision 2026"),
             mpatches.Patch(color="#22d3ee",alpha=0.3,label="Intervalle 95%"),
             mpatches.Patch(color="#ef4444",label="Observation atypique")]
    ax.legend(handles=handles,loc="upper left",fontsize=7.5,
              framealpha=0.15,labelcolor="white")
    fig.tight_layout(); return fig

def fig_famille(fam_results):
    if not fam_results: return None
    fams=list(fam_results.keys())
    fig,ax=plt.subplots(figsize=(11,5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
    for i,fam in enumerate(fams):
        vals=[f["prevision_sc"] for f in fam_results[fam]["fcs"]]
        ax.plot(range(12),vals,marker="o",markersize=4,
                color=FAM_COLORS[i%len(FAM_COLORS)],lw=2,label=fam)
    ax.set_xticks(range(12)); ax.set_xticklabels(MOIS_FR,fontsize=8,color="#94a3b8")
    ax.spines[["top","right","left","bottom"]].set_color(GRID)
    ax.tick_params(colors="#94a3b8",labelsize=8)
    ax.yaxis.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
    ax.set_ylabel("kMAD",color="#94a3b8",fontsize=9)
    ax.legend(loc="upper left",fontsize=8,framealpha=0.15,labelcolor="white")
    fig.tight_layout(); return fig

# ══════════════════════════════════════════════════════════════════════
# ÉCRAN 1 — IMPORT
# ══════════════════════════════════════════════════════════════════════
def screen1():
    nav_bar()
    st.markdown('<div class="el">ÉTAPE 1 / 7</div>',unsafe_allow_html=True)
    st.markdown('<div class="et">Importez votre fichier de ventes historiques</div>',unsafe_allow_html=True)
    st.markdown('<div class="card2"><div class="dz">📄<br><br>Glissez votre fichier <b>Excel ou CSV</b> ici</div></div>',unsafe_allow_html=True)
    up=st.file_uploader("Parcourir…",type=["xlsx","csv"],label_visibility="collapsed")
    st.markdown('<div class="info">ℹ Formats : <b>.xlsx</b> et <b>.csv</b> — 60 observations attendues (Jan 2021 – Déc 2025)</div>',unsafe_allow_html=True)
    if up:
        df,df_fam,err=load_file(up)
        if err:
            st.markdown(f'<div class="er">⊘ <b>Format incorrect</b> — {err}</div>',unsafe_allow_html=True)
            st.session_state.valid=False
        else:
            st.session_state.df=df; st.session_state.df_fam=df_fam; st.session_state.valid=True
            fam_msg=" | Feuille 'Ventes par famille' détectée ✅" if df_fam is not None else ""
            st.markdown(f'<div class="ok">✓ <b>Fichier valide</b> — {len(df)} observations chargées, aucune valeur manquante.{fam_msg}</div>',unsafe_allow_html=True)
            st.markdown('<div class="er">⊘ <b>Format incorrect</b> — Vérifiez que votre fichier contient : Année, Mois, Ventes (kMAD), sans valeur manquante.</div>',unsafe_allow_html=True)
            st.dataframe(df.rename(columns={"annee":"Année","mois":"Mois","ventes":"Ventes (kMAD)"}).head(5),
                         use_container_width=True,hide_index=True)
    if st.session_state.valid:
        if st.button("→ Lancer l'analyse"): go(2)

# ══════════════════════════════════════════════════════════════════════
# ÉCRAN 2 — PARAMÈTRES + TEST AUTOMATIQUE
# ══════════════════════════════════════════════════════════════════════
def screen2():
    nav_bar()
    st.markdown('<div class="el">ÉTAPE 2 / 7</div>',unsafe_allow_html=True)
    st.markdown('<div class="et">Configurez les paramètres de prévision</div>',unsafe_allow_html=True)

    # Calcul CV pour test automatique
    series=st.session_state.df["ventes"].values
    cv=cv_test(series)
    modele_auto="Multiplicatif" if cv>0.3 else "Additif"

    col1,col2=st.columns(2)
    with col1:
        st.markdown('<div class="card2"><b>Méthode de tendance</b>', unsafe_allow_html=True)
        method=st.radio("Méthode",["Moindres carrés (MCO)","Moyennes mobiles (MM)"],
                        label_visibility="collapsed")
        st.caption("Les deux sont calculées et leurs résultats comparés automatiquement.")
        st.markdown('</div>',unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card2"><b>Modèle de saisonnalité</b>',unsafe_allow_html=True)
        model=st.radio("Modèle",["Multiplicatif","Additif"],label_visibility="collapsed")
        badge_m='<span class="badge-g">Recommandé</span>' if model=="Multiplicatif" else '<span class="badge-r">Non recommandé</span>'
        st.markdown(f'Modèle sélectionné : <b>{model}</b>{badge_m}',unsafe_allow_html=True)
        st.caption(f"Test automatique : CV = {cv:.2f} — le modèle {modele_auto.lower()} est adapté.")
        st.markdown('</div>',unsafe_allow_html=True)

    # Test automatique résultats
    with st.spinner("Calcul du test automatique…"):
        r_mult=forecast(st.session_state.df,"MCO","Multiplicatif")
        r_addi=forecast(st.session_state.df,"MCO","Additif")

    st.markdown('<div class="card2"><b>Résultat du test automatique</b>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        badge="✅" if r_mult["rmse"]<=r_addi["rmse"] else "⚠️"
        st.markdown(f"Modèle multiplicatif — RMSE")
        st.markdown(f'<span style="font-size:24px;font-weight:900;color:#60a5fa">{r_mult["rmse"]:.0f} kMAD</span> {badge}',unsafe_allow_html=True)
    with c2:
        badge2="✅" if r_addi["rmse"]<r_mult["rmse"] else "❌"
        st.markdown(f"Modèle additif — RMSE")
        st.markdown(f'<span style="font-size:24px;font-weight:900;color:#f87171">{r_addi["rmse"]:.0f} kMAD</span> {badge2}',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="info">ℹ Le modèle multiplicatif est imposé par le cahier des charges (§6.3) : les pics saisonniers de PAPETIS sont proportionnels au niveau des ventes.</div>',unsafe_allow_html=True)

    c1,c2=st.columns([1,2])
    with c1:
        if st.button("← Retour"): go(1)
    with c2:
        if st.button("📊 Calculer les prévisions"):
            m="MCO" if "MCO" in method else "MM"
            mo=model
            st.session_state.method=m; st.session_state.model=mo
            with st.spinner("Calcul en cours…"):
                r_mco=forecast(st.session_state.df,"MCO",mo)
                r_mm=forecast(st.session_state.df,"MM",mo)
                r=r_mco if m=="MCO" else r_mm
                fam_r={}
                if st.session_state.df_fam is not None:
                    fam_r=forecast_famille(st.session_state.df_fam,m,mo)
            st.session_state.result_mco=r_mco; st.session_state.result_mm=r_mm
            st.session_state.result=r; st.session_state.fam_results=fam_r
            st.session_state.seasonal=r["seas"]; st.session_state.outliers=r["outs"]
            st.session_state.trend_eq=r["eq"]; st.session_state.forecasts=r["fcs"]
            st.session_state.coeff_aout=r["seas"][7]; st.session_state.coeff_mai=r["seas"][4]
            st.session_state.cv=cv
            st.session_state.fig_comp=fig_comparaison(st.session_state.df,r_mco,r_mm)
            st.session_state.fig_results=fig_results(st.session_state.df,r)
            st.session_state.fig_fam=fig_famille(fam_r) if fam_r else None
            go(3)

# ══════════════════════════════════════════════════════════════════════
# ÉCRAN 3 — COMPARAISON MCO VS MM
# ══════════════════════════════════════════════════════════════════════
def screen3():
    nav_bar()
    st.markdown('<div class="el">ÉTAPE 3 / 7</div>',unsafe_allow_html=True)
    st.markdown('<div class="et">Comparaison des méthodes de prévision</div>',unsafe_allow_html=True)

    r_mco=st.session_state.result_mco; r_mm=st.session_state.result_mm
    best_mco=r_mco["mape"]<=r_mm["mape"]

    col1,col2=st.columns(2)
    with col1:
        st.markdown(f'<div class="card2">🔵 <b>Moindres Carrés (MCO)</b>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        c1.metric("MAE",f"{r_mco['mae']:.0f}"); c2.metric("RMSE",f"{r_mco['rmse']:.0f}"); c3.metric("MAPE",f"{r_mco['mape']:.1f}%")
        if best_mco:
            st.markdown('<div class="ok">✅ Méthode retenue pour PAPETIS</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="wa">⚠ Erreur plus élevée</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card2">🟡 <b>Moyennes Mobiles (MM)</b>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        c1.metric("MAE",f"{r_mm['mae']:.0f}"); c2.metric("RMSE",f"{r_mm['rmse']:.0f}"); c3.metric("MAPE",f"{r_mm['mape']:.1f}%")
        if not best_mco:
            st.markdown('<div class="ok">✅ Méthode retenue pour PAPETIS</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="er">▲ Erreur plus élevée</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="card2"><small style="color:#64748b">Graphique comparatif — Historique vs prévisions des deux méthodes</small></div>',unsafe_allow_html=True)
    st.pyplot(st.session_state.fig_comp,use_container_width=True)

    c1,c2=st.columns([1,2])
    with c1:
        if st.button("← Retour"): go(2)
    with c2:
        meilleure="MCO" if best_mco else "MM"
        if st.button(f"→ Utiliser {meilleure} → Résultats"): go(4)

# ══════════════════════════════════════════════════════════════════════
# ÉCRAN 4 — RÉSULTATS + OBSERVATIONS ATYPIQUES + IC
# ══════════════════════════════════════════════════════════════════════
def screen4():
    nav_bar()
    st.markdown('<div class="el">ÉTAPE 4 / 7</div>',unsafe_allow_html=True)
    st.markdown('<div class="et">Prévisions mensuelles — Janvier à Décembre 2026</div>',unsafe_allow_html=True)

    # KPIs
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi"><div class="kpi-l">Tendance {st.session_state.method}</div><div class="kpi-v">{st.session_state.trend_eq}</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi"><div class="kpi-l">Coeff. août (pic)</div><div class="kpi-v">× {st.session_state.coeff_aout:.2f}</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi"><div class="kpi-l">Coeff. mai (creux)</div><div class="kpi-v">× {st.session_state.coeff_mai:.2f}</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # Outliers
    if st.session_state.outliers:
        df=st.session_state.df; lines=""
        for (idx,val,ec,ra) in st.session_state.outliers:
            ml=MOIS_FR[df["mois"].iloc[idx]-1]; an=df["annee"].iloc[idx]
            lines+=f"— {ml} {an} : {val:.0f} kMAD (écart {ec:+.0f} kMAD, ratio × {ra:.2f})<br>"
        st.markdown(f'<div class="wa">⚠ <b>{len(st.session_state.outliers)} observations atypiques détectées</b><br>{lines}Seuil : 2 écarts-types — conservées mais signalées.</div>',unsafe_allow_html=True)

    st.pyplot(st.session_state.fig_results,use_container_width=True)

    st.markdown("#### Tableau des prévisions avec intervalles de confiance 95%")
    fc_df=pd.DataFrame(st.session_state.forecasts)[["label","tendance","coeff","prevision","ic_bas","ic_haut"]]
    fc_df.columns=["Mois","Tendance (kMAD)","Coeff. saisonnier","Prévision (kMAD)","IC bas 95%","IC haut 95%"]
    st.dataframe(fc_df.round(1),use_container_width=True,hide_index=True)

    st.markdown("#### Coefficients saisonniers")
    s=st.session_state.seasonal
    st.dataframe(pd.DataFrame({"Mois":MOIS_FR,"Coeff.":[round(x,3) for x in s]}).T,
                 use_container_width=True,hide_index=True)

    c1,c2=st.columns([1,2])
    with c1:
        if st.button("← Retour"): go(3)
    with c2:
        if st.button("→ Voir les scénarios"): go(5)

# ══════════════════════════════════════════════════════════════════════
# ÉCRAN 5 — SCÉNARIOS
# ══════════════════════════════════════════════════════════════════════
def screen5():
    nav_bar()
    st.markdown('<div class="el">ÉTAPE 5 / 7</div>',unsafe_allow_html=True)
    st.markdown('<div class="et">Simulez différentes hypothèses de croissance</div>',unsafe_allow_html=True)

    st.markdown('<div class="card2"><b>Paramètres des scénarios</b>',unsafe_allow_html=True)
    pes=st.slider("Croissance pessimiste (%)",-30,0,-10,step=5)
    cen=0
    opt=st.slider("Croissance optimiste (%)",0,30,15,step=5)
    st.markdown('</div>',unsafe_allow_html=True)

    with st.spinner("Calcul des 3 scénarios…"):
        r_p=forecast(st.session_state.df,st.session_state.method,st.session_state.model,float(pes))
        r_c=forecast(st.session_state.df,st.session_state.method,st.session_state.model,0.0)
        r_o=forecast(st.session_state.df,st.session_state.method,st.session_state.model,float(opt))

    ca_p=sum(f["prevision_sc"] for f in r_p["fcs"])
    ca_c=sum(f["prevision_sc"] for f in r_c["fcs"])
    ca_o=sum(f["prevision_sc"] for f in r_o["fcs"])

    # Scénario pessimiste
    st.markdown(f'''<div class="sc-card" style="border-left-color:#ef4444">
    <div class="sc-title">📉 Scénario Pessimiste ({pes}%)</div>
    <div class="sc-val" style="color:#ef4444">CA prévu 2026 : {ca_p:,.0f} kMAD</div>
    <div class="sc-desc">Hypothèse : baisse liée à une concurrence accrue ou météo défavorable à la rentrée</div>
    </div>''',unsafe_allow_html=True)

    # Scénario central
    st.markdown(f'''<div class="sc-card" style="border-left-color:#3b82f6">
    <div class="sc-title">📊 Scénario Central (tendance {st.session_state.method})</div>
    <div class="sc-val" style="color:#3b82f6">CA prévu 2026 : {ca_c:,.0f} kMAD</div>
    <div class="sc-desc">Hypothèse : poursuite de la tendance historique {st.session_state.trend_eq}</div>
    </div>''',unsafe_allow_html=True)

    # Scénario optimiste
    st.markdown(f'''<div class="sc-card" style="border-left-color:#22c55e">
    <div class="sc-title">📈 Scénario Optimiste (+{opt}%)</div>
    <div class="sc-val" style="color:#22c55e">CA prévu 2026 : {ca_o:,.0f} kMAD</div>
    <div class="sc-desc">Hypothèse : gain de nouveaux clients libraires ou hausse de la demande scolaire</div>
    </div>''',unsafe_allow_html=True)

    # Tableau comparatif 3 scénarios
    st.markdown("#### Tableau comparatif mensuel")
    sc_df=pd.DataFrame({
        "Mois":         [f["label"] for f in r_c["fcs"]],
        f"Pessimiste ({pes}%) kMAD": [round(f["prevision_sc"],0) for f in r_p["fcs"]],
        "Central (kMAD)":            [round(f["prevision_sc"],0) for f in r_c["fcs"]],
        f"Optimiste (+{opt}%) kMAD": [round(f["prevision_sc"],0) for f in r_o["fcs"]],
    })
    st.dataframe(sc_df,use_container_width=True,hide_index=True)

    # Stocker pour export
    st.session_state["sc_df"]=sc_df
    st.session_state["ca_c"]=ca_c; st.session_state["ca_p"]=ca_p; st.session_state["ca_o"]=ca_o
    st.session_state["pes"]=pes; st.session_state["opt"]=opt

    c1,c2=st.columns([1,2])
    with c1:
        if st.button("← Retour"): go(4)
    with c2:
        if st.button("→ Analyse par famille"): go(6)

# ══════════════════════════════════════════════════════════════════════
# ÉCRAN 6 — FAMILLES
# ══════════════════════════════════════════════════════════════════════
def screen6():
    nav_bar()
    st.markdown('<div class="el">ÉTAPE 6 / 7</div>',unsafe_allow_html=True)
    st.markdown('<div class="et">Prévisions 2026 par famille de produits</div>',unsafe_allow_html=True)

    fam_r=st.session_state.fam_results
    if not fam_r:
        st.markdown('<div class="wa">⚠ Feuille "Ventes par famille" non trouvée dans votre fichier. Réimportez un fichier Excel avec cette feuille.</div>',unsafe_allow_html=True)
        if st.button("← Retour"): go(5)
        return

    fams=list(fam_r.keys())
    ca_fam={fam:sum(f["prevision_sc"] for f in fam_r[fam]["fcs"]) for fam in fams}
    total_ca=sum(ca_fam.values())

    col1,col2=st.columns(2)
    with col1:
        st.markdown('<div class="card2"><b>Répartition du CA prévu 2026</b><br><br>',unsafe_allow_html=True)
        max_ca=max(ca_fam.values())
        for i,fam in enumerate(fams):
            val=ca_fam[fam]; pct=val/total_ca*100
            bar_w=int(val/max_ca*100)
            color=FAM_COLORS[i%len(FAM_COLORS)]
            st.markdown(f'''<div class="bar-wrap">
            <span class="bar-label">{fam}</span>
            <div class="bar-outer"><div class="bar-inner" style="width:{bar_w}%;background:{color}"></div></div>
            <span class="bar-val">{val:,.0f} kMAD</span></div>''',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card2"><b>Coefficients saisonniers août par famille</b><br><br>',unsafe_allow_html=True)
        # En-tête
        st.markdown('<div style="display:flex;gap:10px;font-size:11px;color:#64748b;padding-bottom:6px;border-bottom:1px solid #2d3f55;margin-bottom:8px"><span style="flex:1">Famille</span><span style="width:80px;text-align:center">Coeff. août</span><span style="width:80px;text-align:center">Niveau</span></div>',unsafe_allow_html=True)
        for fam in fams:
            coeff=fam_r[fam]["seas"][7]
            if coeff>=2.5: niveau,col="Fort pic","#ef4444"
            elif coeff>=1.8: niveau,col="Pic","#f59e0b"
            elif coeff>=1.3: niveau,col="Modéré","#a78bfa"
            else: niveau,col="Stable","#22c55e"
            st.markdown(f'<div style="display:flex;gap:10px;align-items:center;padding:5px 0;border-bottom:1px solid #1e2d45"><span style="flex:1;font-size:13px;color:#e2e8f0">{fam}</span><span style="width:80px;text-align:center;font-size:15px;font-weight:800;color:#60a5fa">{coeff:.1f}</span><span style="width:80px;text-align:center"><span style="background:{col}22;color:{col};border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700">{niveau}</span></span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # Graphique
    if st.session_state.fig_fam:
        st.pyplot(st.session_state.fig_fam,use_container_width=True)

    # Note info
    fams_sorted=sorted(ca_fam,key=ca_fam.get,reverse=True)
    top2=fams_sorted[:2]
    top2_pct=sum(ca_fam[f] for f in top2)/total_ca*100
    st.markdown(f'<div class="info">ℹ {" + ".join(top2)} = {top2_pct:.0f}% du CA. Ce sont les familles prioritaires pour la gestion des stocks et des approvisionnements.</div>',unsafe_allow_html=True)

    # Tableau
    fam_rows=[{"Famille":fam,**{f["label"]:round(f["prevision_sc"],1) for f in fam_r[fam]["fcs"]}}
              for fam in fams]
    st.dataframe(pd.DataFrame(fam_rows),use_container_width=True,hide_index=True)

    c1,c2=st.columns([1,2])
    with c1:
        if st.button("← Retour"): go(5)
    with c2:
        if st.button("→ Exporter les résultats"): go(7)

# ══════════════════════════════════════════════════════════════════════
# EXPORT PDF
# ══════════════════════════════════════════════════════════════════════
def gen_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image as RI
    from reportlab.lib.units import cm
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,
                          leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    sty=getSampleStyleSheet()
    T1=ParagraphStyle("T1",parent=sty["Heading1"],fontSize=16,
                      textColor=colors.HexColor("#1d4ed8"),spaceAfter=4)
    T2=ParagraphStyle("T2",parent=sty["Heading2"],fontSize=12,
                      textColor=colors.HexColor("#1e40af"),spaceAfter=4)
    BD=sty["BodyText"]
    story=[]
    story.append(Paragraph("PAPETIS DISTRIBUTION SARL",T1))
    story.append(Paragraph("Rapport de prévision mensuelle des ventes — 2026",T2))
    story.append(Paragraph(
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} | "
        f"Méthode : {st.session_state.method} | Modèle : {st.session_state.model}",BD))
    story.append(Spacer(1,0.3*cm))

    # Comparaison méthodes
    story.append(Paragraph("1. Comparaison des méthodes",T2))
    r_mco=st.session_state.result_mco; r_mm=st.session_state.result_mm
    err_d=[["Méthode","MAE (kMAD)","RMSE (kMAD)","MAPE (%)"],
           ["MCO",f"{r_mco['mae']:.0f}",f"{r_mco['rmse']:.0f}",f"{r_mco['mape']:.1f}%"],
           ["MM12",f"{r_mm['mae']:.0f}",f"{r_mm['rmse']:.0f}",f"{r_mm['mape']:.1f}%"]]
    t=Table(err_d,colWidths=[5*cm,3.5*cm,3.5*cm,3.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff6ff")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cbd5e1")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
    ]))
    story.append(t); story.append(Spacer(1,0.3*cm))

    # Tendance
    story.append(Paragraph("2. Tendance et saisonnalité",T2))
    story.append(Paragraph(f"Équation : <b>{st.session_state.trend_eq}</b>",BD))
    story.append(Paragraph(f"Coeff. août (pic rentrée) : <b>× {st.session_state.coeff_aout:.2f}</b>",BD))
    story.append(Paragraph(f"Coeff. mai (creux) : <b>× {st.session_state.coeff_mai:.2f}</b>",BD))
    story.append(Spacer(1,0.3*cm))

    # Observations atypiques
    story.append(Paragraph("3. Observations atypiques",T2))
    df=st.session_state.df
    for (idx,val,ec,ra) in st.session_state.outliers:
        ml=MOIS_FR[df["mois"].iloc[idx]-1]; an=df["annee"].iloc[idx]
        story.append(Paragraph(f"• {ml} {an} : {val:.0f} kMAD — écart {ec:+.0f} kMAD (ratio × {ra:.2f}). Conservée.",BD))
    story.append(Spacer(1,0.3*cm))

    # Graphique résultats
    story.append(Paragraph("4. Graphique historique + prévisions 2026",T2))
    ib=io.BytesIO()
    st.session_state.fig_results.savefig(ib,format="png",dpi=150,
                                          bbox_inches="tight",facecolor="#0f1623")
    ib.seek(0); story.append(RI(ib,width=16*cm,height=7*cm))
    story.append(Spacer(1,0.3*cm))

    # Tableau prévisions
    story.append(Paragraph("5. Prévisions détaillées avec intervalles de confiance",T2))
    hdr=["Mois","Tendance","Coeff.","Prévision","IC bas","IC haut"]
    data=[hdr]+[[f["label"],f"{f['tendance']:.0f}",f"{f['coeff']:.3f}",
                 f"{f['prevision']:.0f}",f"{f['ic_bas']:.0f}",f"{f['ic_haut']:.0f}"]
                for f in st.session_state.forecasts]
    t2=Table(data,colWidths=[3.5*cm,3*cm,2.5*cm,3*cm,2.5*cm,2.5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff6ff")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cbd5e1")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
    ]))
    story.append(t2); story.append(Spacer(1,0.3*cm))

    # Scénarios
    if "sc_df" in st.session_state and st.session_state["sc_df"] is not None:
        story.append(Paragraph("6. Analyse de scénarios",T2))
        ca_p=st.session_state.get("ca_p",0); ca_c=st.session_state.get("ca_c",0)
        ca_o=st.session_state.get("ca_o",0); pes=st.session_state.get("pes",-10)
        opt=st.session_state.get("opt",15)
        for lbl,val,desc in [
            (f"Pessimiste ({pes}%)",ca_p,"Concurrence accrue ou rentrée défavorable"),
            ("Central",ca_c,f"Poursuite de la tendance {st.session_state.trend_eq}"),
            (f"Optimiste (+{opt}%)",ca_o,"Nouveaux clients ou hausse de la demande scolaire"),
        ]:
            story.append(Paragraph(f"• <b>{lbl}</b> : CA prévu = {val:,.0f} kMAD — {desc}",BD))
        story.append(Spacer(1,0.3*cm))

    # Recommandations
    story.append(Paragraph("7. Recommandations opérationnelles",T2))
    for r in [
        f"Passer les commandes en mars–avril pour la rentrée août–septembre (pic × {st.session_state.coeff_aout:.1f}).",
        "Réduire les commandes pour mai–juin (creux saisonnier).",
        "Utiliser le scénario pessimiste pour le stock de sécurité, l'optimiste pour les commandes maximales.",
        "Prioriser les familles Cahiers et Classeurs pour les approvisionnements.",
    ]:
        story.append(Paragraph(f"• {r}",BD))

    doc.build(story); buf.seek(0); return buf.read()

# ══════════════════════════════════════════════════════════════════════
# ÉCRAN 7 — EXPORT
# ══════════════════════════════════════════════════════════════════════
def screen7():
    nav_bar()
    st.markdown('<div class="el">ÉTAPE 7 / 7</div>',unsafe_allow_html=True)
    st.markdown('<div class="et">Exportez vos résultats de prévision</div>',unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.markdown('''<div class="card2" style="text-align:center">
        <div style="font-size:48px;color:#22c55e">📗</div>
        <h4>Fichier Excel (.xlsx)</h4>
        <p>Prévisions 2026, tendance MCO, 12 coefficients saisonniers, 3 scénarios, observations atypiques signalées.</p>
        </div>''',unsafe_allow_html=True)

        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as wr:
            # Prévisions + IC
            pd.DataFrame(st.session_state.forecasts)[
                ["label","tendance","coeff","prevision","ic_bas","ic_haut"]
            ].rename(columns={"label":"Mois","tendance":"Tendance (kMAD)",
                               "coeff":"Coeff. saisonnier","prevision":"Prévision (kMAD)",
                               "ic_bas":"IC bas 95%","ic_haut":"IC haut 95%"}
            ).round(1).to_excel(wr,sheet_name="Prévisions 2026",index=False)
            # Scénarios
            if "sc_df" in st.session_state and st.session_state["sc_df"] is not None:
                st.session_state["sc_df"].to_excel(wr,sheet_name="Scénarios",index=False)
            # Coefficients
            s=st.session_state.seasonal
            pd.DataFrame({"Mois":MOIS_FR,"Coeff. saisonnier":s}).to_excel(
                wr,sheet_name="Coefficients saisonniers",index=False)
            # Erreurs
            r_mco=st.session_state.result_mco; r_mm=st.session_state.result_mm
            pd.DataFrame({"Méthode":["MCO","MM12"],
                "MAE":[round(r_mco["mae"],1),round(r_mm["mae"],1)],
                "RMSE":[round(r_mco["rmse"],1),round(r_mm["rmse"],1)],
                "MAPE (%)":[round(r_mco["mape"],2),round(r_mm["mape"],2)]
            }).to_excel(wr,sheet_name="Indicateurs erreur",index=False)
            # Outliers
            df=st.session_state.df
            if st.session_state.outliers:
                pd.DataFrame([{"Mois":MOIS_FR[df["mois"].iloc[i]-1],"Année":df["annee"].iloc[i],
                    "Ventes (kMAD)":v,"Écart":e,"Ratio":round(r,2)}
                    for i,v,e,r in st.session_state.outliers
                ]).to_excel(wr,sheet_name="Observations atypiques",index=False)
            # Familles
            if st.session_state.fam_results:
                rows=[{"Famille":fam,**{f["label"]:round(f["prevision_sc"],1)
                       for f in res["fcs"]}}
                      for fam,res in st.session_state.fam_results.items()]
                pd.DataFrame(rows).to_excel(wr,sheet_name="Prévisions par famille",index=False)
            # Historique
            df.rename(columns={"annee":"Année","mois":"Mois","ventes":"Ventes (kMAD)"}
            ).to_excel(wr,sheet_name="Historique",index=False)

        st.download_button("⬇ Télécharger .xlsx",data=buf.getvalue(),
                           file_name="PAPETIS_previsions_2026.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    with col2:
        st.markdown('''<div class="card2" style="text-align:center">
        <div style="font-size:48px;color:#ef4444">📄</div>
        <h4>Rapport PDF</h4>
        <p>Rapport complet : graphique, analyse saisonnière, scénarios, décomposition par famille et recommandations de stock.</p>
        </div>''',unsafe_allow_html=True)
        st.download_button("⬇ Télécharger .pdf",data=gen_pdf(),
                           file_name="PAPETIS_rapport_previsions_2026.pdf",
                           mime="application/pdf",use_container_width=True)

    eq=st.session_state.trend_eq
    st.markdown(
        f'<div class="ok">✓ <b>Analyse terminée</b> — Prévisions Jan–Déc 2026 générées '
        f'(modèle {st.session_state.model}, {st.session_state.method} : {eq})</div>',
        unsafe_allow_html=True)

    if st.button("← Retour"): go(6)

# ══════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════
{1:screen1,2:screen2,3:screen3,4:screen4,5:screen5,6:screen6,7:screen7}[st.session_state.step]()
