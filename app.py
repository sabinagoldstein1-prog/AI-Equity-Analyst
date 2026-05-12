"""AI Equity Analyst v4 - No chatbot, fixed data"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import (fetch_prices, fetch_fundamentals, run_ml,
                    run_predictive_model, run_trading_system,
                    run_monte_carlo, run_scoring, PERFIS)

st.set_page_config(page_title="AI Equity Analyst", page_icon="\U0001f4b9", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
:root {--bg:#0a0e17;--card:#111827;--border:#1e2d3d;--green:#00d26a;--red:#ff3b3b;--blue:#0ea5e9;--gold:#facc15;--text:#e2e8f0;--muted:#64748b;}
.stApp {background-color:var(--bg)!important;}
.block-container {padding-top:1rem;max-width:1400px;}
h1,h2,h3 {font-family:'Inter',sans-serif!important;color:var(--blue)!important;}
.stMetric {background:var(--card);border:1px solid var(--border);border-radius:6px;padding:10px 14px;}
.stMetric label {font-family:'JetBrains Mono',monospace!important;font-size:11px!important;color:var(--muted)!important;text-transform:uppercase;letter-spacing:1px;}
.stMetric [data-testid="stMetricValue"] {font-family:'JetBrains Mono',monospace!important;font-size:22px!important;}
div[data-testid="stSidebar"] {background:#070b12!important;border-right:1px solid var(--border);}
.stTabs [data-baseweb="tab"] {font-family:'Inter',sans-serif;font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;}
.stButton>button[kind="primary"] {background:linear-gradient(135deg,#0ea5e9,#3b82f6);border:none;font-weight:700;}
</style>""", unsafe_allow_html=True)

PL = dict(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#111827",
          font=dict(family="JetBrains Mono,monospace",size=11,color="#e2e8f0"),margin=dict(l=40,r=20,t=50,b=40))

def fmt(t): return str(t).replace(".SA","")
def fmt_pct(v): return f"{v*100:.1f}%" if pd.notna(v) else "-"
def fmt_x(v): return f"{v:.1f}x" if pd.notna(v) else "-"
def fmt_brl(v): return f"R${v:.2f}" if pd.notna(v) else "-"

with st.sidebar:
    st.markdown("## \U0001f4b9 AI EQUITY ANALYST")
    st.caption("QUANT RESEARCH TERMINAL | FGV 2026")
    st.markdown("---")
    tickers_input = st.text_area("UNIVERSE",value="PETR4\nITUB4\nBBDC4\nBBAS3\nVALE3\nWEGE3\nABEV3\nPRIO3\nEGIE3\nEQTL3",height=180)
    perfil = st.selectbox("STRATEGY",["conservador","moderado","agressivo"],index=1)
    data_inicio = st.date_input("LOOKBACK",value=pd.Timestamp("2021-01-01"))
    st.markdown("---")
    p = PERFIS[perfil]
    st.code(f"WEIGHTS [{perfil.upper()}]\nMKT  = {p['mercado']:.0%}\nQUAL = {p['qualidade']:.0%}",language="yaml")
    run_btn = st.button("\U0001f680 RUN ANALYSIS",type="primary",use_container_width=True)

def parse_tickers(raw):
    t = [x.strip().upper() for x in raw.replace(",", " ").replace("\n", " ").split() if x.strip()]
    return [x if x.endswith(".SA") else x+".SA" for x in t]

keys = ["result","prices","fund","ml_metrics","ml_fi","ml_preds","ml_clusters","pred_metrics","pred_fi","trading_curves","trading_summary","mc_carteiras","mc_best","mc_tickers"]
for k in keys:
    if k not in st.session_state: st.session_state[k] = None

if run_btn:
    tickers = parse_tickers(tickers_input)
    if len(tickers)<2: st.error("MIN 2 TICKERS"); st.stop()
    progress = st.progress(0,"INITIALIZING...")
    try:
        progress.progress(10,"FETCHING PRICES...")
        st.session_state.prices = fetch_prices(tickers,str(data_inicio))
        tickers_ok = st.session_state.prices["ticker"].unique().tolist()
        progress.progress(30,"FETCHING FUNDAMENTALS...")
        st.session_state.fund = fetch_fundamentals(tickers_ok)
        progress.progress(45,"TRAINING ML MODEL...")
        ml = run_ml(st.session_state.prices)
        st.session_state.ml_metrics,st.session_state.ml_fi,st.session_state.ml_preds,st.session_state.ml_clusters = ml
        progress.progress(60,"PREDICTIVE MODEL...")
        st.session_state.pred_metrics,st.session_state.pred_fi = run_predictive_model(st.session_state.prices)
        progress.progress(75,"BACKTESTING...")
        tc,ts = run_trading_system(st.session_state.prices)
        st.session_state.trading_curves = tc; st.session_state.trading_summary = ts
        progress.progress(88,"MONTE CARLO...")
        mc,best,mct = run_monte_carlo(st.session_state.prices)
        st.session_state.mc_carteiras = mc; st.session_state.mc_best = best; st.session_state.mc_tickers = mct
        progress.progress(95,"SCORING...")
        st.session_state.result = run_scoring(st.session_state.prices,st.session_state.fund,perfil)
        progress.progress(100,"COMPLETE")
    except Exception as e:
        st.error(f"ERROR: {e}"); import traceback; st.code(traceback.format_exc()); st.stop()

if st.session_state.result is None:
    st.markdown("# \U0001f4b9 AI EQUITY ANALYST\n### Quantitative Research Terminal\n\n"
                "```\nSYSTEM STATUS: READY\nENGINE: 7 tools | RF Walk-Forward | Monte Carlo 10K\n```\n\n"
                "1. Input tickers  2. Select profile  3. Click **RUN ANALYSIS**\n\n*FGV 2026*")
    st.stop()

result=st.session_state.result; prices=st.session_state.prices; fund=st.session_state.fund

tab1,tab2,tab3,tab4,tab5 = st.tabs(["RANKING","MARKET","ML ENGINE","SIGNALS","PORTFOLIO"])

# === RANKING ===
with tab1:
    st.markdown("### COMPOSITE RANKING")
    nb=sum("BUY" in str(r) for r in result["recomendacao"]); nh=sum("HOLD" in str(r) for r in result["recomendacao"]); ns=sum("SELL" in str(r) for r in result["recomendacao"])
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("UNIVERSE",len(result)); c2.metric("BUY",nb); c3.metric("HOLD",nh); c4.metric("SELL",ns); c5.metric("AVG",f"{result['score'].mean():.0f}")
    r=result.sort_values("score",ascending=True)
    colors=["#00d26a" if "BUY" in rec else "#ff3b3b" if "SELL" in rec else "#facc15" for rec in r["recomendacao"]]
    fig=go.Figure(go.Bar(y=r["ticker"].apply(fmt),x=r["score"],orientation="h",marker_color=colors,
        text=[f"{s:.0f} | {rec}" for s,rec in zip(r["score"],r["recomendacao"])],textposition="outside",textfont=dict(family="JetBrains Mono",size=11)))
    fig.add_vline(x=50,line_dash="dash",line_color="#64748b",opacity=0.5)
    fig.update_layout(title="COMPOSITE SCORE",xaxis_range=[0,115],height=max(350,len(r)*42),**PL)
    st.plotly_chart(fig,use_container_width=True)
    # Table with formatted values
    tbl=result[["rank","ticker","nome","setor","preco","P_L","P_VP","EV_EBITDA","score","recomendacao"]].sort_values("rank").copy()
    tbl["ticker"]=tbl["ticker"].apply(fmt)
    tbl["preco"]=tbl["preco"].apply(fmt_brl)
    tbl["P_L"]=tbl["P_L"].apply(fmt_x)
    tbl["P_VP"]=tbl["P_VP"].apply(fmt_x)
    tbl["EV_EBITDA"]=tbl["EV_EBITDA"].apply(fmt_x)
    tbl["nome"]=tbl["nome"].fillna("-")
    tbl["setor"]=tbl["setor"].fillna("-")
    tbl.columns=["#","TICKER","NAME","SECTOR","PRICE","P/E","P/B","EV/EBITDA","SCORE","REC"]
    st.dataframe(tbl,use_container_width=True,hide_index=True)

# === MARKET ===
with tab2:
    st.markdown("### MARKET DASHBOARD")
    col1,col2=st.columns(2)
    with col1:
        d=result.dropna(subset=["vol_21","mom_6m","score"])
        fig=px.scatter(d,x=d["vol_21"]*100,y=d["mom_6m"]*100,size="score",color="recomendacao",text=d["ticker"].apply(fmt),
            color_discrete_map={"BUY":"#00d26a","HOLD":"#facc15","SELL":"#ff3b3b"},labels={"x":"VOL 21D (%)","y":"MOM 6M (%)"},title="RISK vs RETURN")
        fig.update_traces(textposition="top center",textfont=dict(size=10,family="JetBrains Mono"))
        fig.update_layout(**PL)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        top5=result.head(5)["ticker"].tolist()
        fig=go.Figure()
        for t in top5:
            sub=prices[prices["ticker"]==t].dropna(subset=["preco"])
            if sub.empty: continue
            norm=sub["preco"]/sub["preco"].iloc[0]*100
            fig.add_trace(go.Scatter(x=sub["data"],y=norm,name=fmt(t),mode="lines"))
        fig.add_hline(y=100,line_dash="dash",line_color="#64748b",opacity=0.4)
        fig.update_layout(title="TOP 5 PERFORMANCE (BASE 100)",yaxis_title="CUMULATIVE (%)",**PL)
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### FUNDAMENTALS")
    fd=fund[["ticker","nome","setor","preco","P_L","P_VP","EV_EBITDA","returnOnEquity","profitMargins","revenueGrowth"]].copy()
    fd["ticker"]=fd["ticker"].apply(fmt); fd["preco"]=fd["preco"].apply(fmt_brl)
    for c in ["P_L","P_VP","EV_EBITDA"]: fd[c]=fd[c].apply(fmt_x)
    for c in ["returnOnEquity","profitMargins","revenueGrowth"]: fd[c]=fd[c].apply(fmt_pct)
    fd["nome"]=fd["nome"].fillna("-"); fd["setor"]=fd["setor"].fillna("-")
    fd.columns=["TICKER","NAME","SECTOR","PRICE","P/E","P/B","EV/EBITDA","ROE","MARGIN","GROWTH"]
    st.dataframe(fd,use_container_width=True,hide_index=True)

# === ML ===
with tab3:
    st.markdown("### ML ENGINE | RANDOM FOREST")
    col1,col2=st.columns(2)
    with col1:
        ml_m=st.session_state.ml_metrics
        if ml_m is not None and not ml_m.empty:
            st.markdown("#### WALK-FORWARD")
            st.dataframe(ml_m.round(4),use_container_width=True,hide_index=True)
            fig=px.bar(ml_m,x="year",y="spearman_ic",title="SPEARMAN IC",color="spearman_ic",color_continuous_scale=["#ff3b3b","#facc15","#00d26a"],color_continuous_midpoint=0)
            fig.update_layout(**PL); st.plotly_chart(fig,use_container_width=True)
        else: st.info("Insufficient data for walk-forward.")
    with col2:
        fi=st.session_state.ml_fi
        if fi is not None and not fi.empty:
            st.markdown("#### FEATURE IMPORTANCE")
            fi_mean=fi.mean().sort_values(ascending=True)
            fig=go.Figure(go.Bar(y=fi_mean.index,x=fi_mean.values,orientation="h",marker_color="#0ea5e9"))
            fig.update_layout(title="AVG IMPORTANCE",**PL); st.plotly_chart(fig,use_container_width=True)
    preds=st.session_state.ml_preds
    if preds is not None and not preds.empty and "pred_ret_12m" in preds.columns:
        st.markdown("#### 12M RETURN FORECAST")
        pp=preds[["ticker","pred_ret_12m","rank_pred"]].dropna().copy()
        pp["ticker"]=pp["ticker"].apply(fmt); pp["pred_ret_12m"]=(pp["pred_ret_12m"]*100).round(1)
        pp=pp.sort_values("rank_pred")
        fig=go.Figure(go.Bar(x=pp["ticker"],y=pp["pred_ret_12m"],marker_color=["#00d26a" if v>0 else "#ff3b3b" for v in pp["pred_ret_12m"]],
            text=[f"{v:+.1f}%" for v in pp["pred_ret_12m"]],textposition="outside"))
        fig.update_layout(title="PREDICTED 12M RETURN (%)",**PL); st.plotly_chart(fig,use_container_width=True)
    clust=st.session_state.ml_clusters
    if clust is not None and not clust.empty and "perfil" in clust.columns:
        st.markdown("#### CLUSTERS")
        cl=clust[["ticker","perfil","vol_21","mom_6m","drawdown"]].copy()
        cl["ticker"]=cl["ticker"].apply(fmt)
        for c in ["vol_21","mom_6m","drawdown"]: cl[c]=cl[c].apply(fmt_pct)
        st.dataframe(cl.sort_values("perfil"),use_container_width=True,hide_index=True)

# === SIGNALS ===
with tab4:
    st.markdown("### SIGNALS")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("#### DAILY DIRECTION FORECAST")
        pm=st.session_state.pred_metrics
        if pm is not None and not pm.empty:
            pm_d=pm.copy(); pm_d["ticker"]=pm_d["ticker"].apply(fmt)
            fig=px.bar(pm_d,x="ticker",y="auc",title="AUC (0.5=random)",color="auc",color_continuous_scale=["#ff3b3b","#facc15","#00d26a"],color_continuous_midpoint=0.5)
            fig.add_hline(y=0.5,line_dash="dash",line_color="#ff3b3b",opacity=0.5)
            fig.update_layout(**PL); st.plotly_chart(fig,use_container_width=True)
        else: st.info("Insufficient data.")
    with col2:
        st.markdown("#### MA CROSSOVER")
        ts=st.session_state.trading_summary
        if ts is not None and not ts.empty:
            ts_d=ts.copy(); ts_d["ticker"]=ts_d["ticker"].apply(fmt)
            fig=go.Figure()
            fig.add_trace(go.Bar(name="BUY & HOLD",x=ts_d["ticker"],y=ts_d["ret_buyhold"],marker_color="#0ea5e9"))
            fig.add_trace(go.Bar(name="MA STRATEGY",x=ts_d["ticker"],y=ts_d["ret_estrategia"],marker_color="#00d26a"))
            fig.update_layout(barmode="group",title="CUMULATIVE RETURN (%)",**PL); st.plotly_chart(fig,use_container_width=True)
    tc=st.session_state.trading_curves; ts=st.session_state.trading_summary
    if tc is not None and not tc.empty and ts is not None and not ts.empty:
        sel=st.selectbox("TICKER:",[fmt(t) for t in ts["ticker"].tolist()])
        sel_sa=sel+".SA" if not sel.endswith(".SA") else sel
        sub=tc[tc["ticker"]==sel_sa]
        if not sub.empty:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=sub["data"],y=sub["acum_acao"],name="BUY & HOLD",line=dict(color="#0ea5e9",width=2)))
            fig.add_trace(go.Scatter(x=sub["data"],y=sub["acum_est"],name="MA STRATEGY",line=dict(color="#00d26a",width=2)))
            fig.update_layout(title=f"EQUITY CURVE: {sel}",**PL); st.plotly_chart(fig,use_container_width=True)

# === PORTFOLIO ===
with tab5:
    st.markdown("### PORTFOLIO OPTIMIZATION | MONTE CARLO")
    mc=st.session_state.mc_carteiras; best=st.session_state.mc_best; mct=st.session_state.mc_tickers
    if mc is not None and not mc.empty and best:
        col1,col2=st.columns([2,1])
        with col1:
            fig=px.scatter(mc,x="risco",y="retorno",color="sharpe",color_continuous_scale="viridis",
                labels={"risco":"ANNUAL RISK","retorno":"ANNUAL RETURN","sharpe":"SHARPE"},title="EFFICIENT FRONTIER | 10K SIMULATIONS")
            fig.add_trace(go.Scatter(x=[best["risco"]],y=[best["retorno"]],mode="markers",
                marker=dict(size=18,color="#ff3b3b",symbol="star",line=dict(width=2,color="white")),name="MAX SHARPE"))
            fig.update_layout(**PL); st.plotly_chart(fig,use_container_width=True)
        with col2:
            st.markdown("#### OPTIMAL PORTFOLIO")
            st.metric("SHARPE",f"{best['sharpe']:.3f}")
            st.metric("RETURN",f"{best['retorno']*100:.1f}%")
            st.metric("RISK",f"{best['risco']*100:.1f}%")
            st.markdown("---")
            st.markdown("#### ALLOCATION")
            alloc=[(fmt(t),best.get(t,0)*100) for t in mct]
            alloc.sort(key=lambda x:-x[1])
            for tk,w in alloc:
                st.markdown(f"`{tk:>6s}` **{w:.1f}%**")
    else: st.info("Insufficient data.")

st.sidebar.markdown("---")
st.sidebar.code("AI EQUITY ANALYST v4.0\nFGV | 2026",language="yaml")
