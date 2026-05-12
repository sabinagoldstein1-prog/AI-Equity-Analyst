"""
engine.py — Motor do Agente de IA (v3)
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
np.random.seed(42)

def safe_div(a, b):
    try:
        a, b = float(a), float(b)
        if np.isnan(a) or np.isnan(b) or b == 0: return np.nan
        return a / b
    except: return np.nan

# === TOOL 1: PRECOS ===
def fetch_prices(tickers, start="2021-01-01"):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series): raw = raw.to_frame(name=tickers[0])
    p = raw.stack(future_stack=True).reset_index()
    p.columns = ["data", "ticker", "preco"]
    p["data"] = pd.to_datetime(p["data"])
    p = p.dropna(subset=["preco"]).sort_values(["ticker", "data"])
    g = p.groupby("ticker")
    p["ret_dia"] = g["preco"].pct_change()
    p["vol_21"] = g["ret_dia"].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    p["mom_6m"] = g["preco"].transform(lambda x: x.pct_change(126))
    p["mom_12m"] = g["preco"].transform(lambda x: x.pct_change(252))
    p["drawdown"] = g["preco"].transform(lambda x: (x - x.cummax()) / x.cummax())
    return p

# === TOOL 2: FUNDAMENTOS ===
def fetch_fundamentals(tickers):
    """Busca fundamentos do Yahoo Finance. Nunca retorna ? ou None nas colunas principais."""
    rows = []
    for t in tickers:
        r = {"ticker": t, "nome": t.replace(".SA",""), "setor": "-",
             "preco": np.nan, "marketCap": np.nan, "shares": np.nan,
             "P_L": np.nan, "P_VP": np.nan, "EV_EBITDA": np.nan,
             "EV": np.nan, "lucro": np.nan, "pl_equity": np.nan,
             "totalDebt": np.nan, "summary": "", "div_yield": np.nan,
             "profitMargins": np.nan, "returnOnEquity": np.nan,
             "revenueGrowth": np.nan, "ebitdaMargins": np.nan,
             "debtToEquity": np.nan}
        try:
            info = yf.Ticker(t).info
            if not info or not isinstance(info, dict):
                rows.append(r)
                continue
            r["nome"] = str(info.get("shortName") or info.get("longName") or t.replace(".SA",""))
            r["setor"] = str(info.get("sector") or info.get("industry") or "-")
            r["preco"] = float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or np.nan)
            r["marketCap"] = float(info.get("marketCap") or np.nan)
            r["shares"] = float(info.get("sharesOutstanding") or info.get("floatShares") or np.nan)
            r["P_L"] = float(info.get("trailingPE") or info.get("forwardPE") or np.nan)
            r["P_VP"] = float(info.get("priceToBook") or np.nan)
            r["EV_EBITDA"] = float(info.get("enterpriseToEbitda") or np.nan)
            r["EV"] = float(info.get("enterpriseValue") or np.nan)
            r["totalDebt"] = float(info.get("totalDebt") or np.nan)
            r["div_yield"] = float(info.get("dividendYield") or np.nan)
            r["profitMargins"] = float(info.get("profitMargins") or np.nan)
            r["returnOnEquity"] = float(info.get("returnOnEquity") or np.nan)
            r["revenueGrowth"] = float(info.get("revenueGrowth") or np.nan)
            r["ebitdaMargins"] = float(info.get("ebitdaMargins") or np.nan)
            r["debtToEquity"] = float(info.get("debtToEquity") or np.nan)
            r["summary"] = str(info.get("longBusinessSummary") or "").lower()
            r["lucro"] = safe_div(r["marketCap"], r["P_L"])
            r["pl_equity"] = safe_div(r["marketCap"], r["P_VP"])
            if pd.isna(r["EV_EBITDA"]) and pd.notna(r["EV"]) and pd.notna(r["lucro"]) and r["lucro"] > 0:
                r["EV_EBITDA"] = safe_div(r["EV"], r["lucro"])
        except Exception:
            pass
        rows.append(r)
    return pd.DataFrame(rows)

# === TOOL 3: ML WALK-FORWARD + CLUSTERING ===
def run_ml(prices):
    feat = prices.copy()
    feat["month"] = feat["data"].dt.to_period("M")
    me = feat.sort_values("data").groupby(["ticker","month"]).tail(1).copy()
    me = me.sort_values(["ticker","data"])
    mkt = ["vol_21","mom_6m","mom_12m","drawdown"]
    for c in mkt:
        me[f"{c}_z"] = me.groupby("data")[c].transform(lambda s: (s-s.mean())/(s.std() if s.std()>0 else 1))
    mkt_z = [f"{c}_z" for c in mkt]
    me["ret_12m_fwd"] = me.groupby("ticker")["preco"].transform(lambda s: s.shift(-12)/s-1.0)
    core = mkt + mkt_z
    data_ml = me.dropna(subset=core+["ret_12m_fwd"]).copy()
    data_ml["year"] = data_ml["data"].dt.year
    yrs = sorted(data_ml["year"].unique())
    metrics, fis = [], []
    for ty in yrs[2:]:
        tr = data_ml[data_ml["year"]<ty]; te = data_ml[data_ml["year"]==ty]
        if len(tr)<20 or len(te)<3: continue
        m = RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=3,random_state=42,n_jobs=-1)
        m.fit(tr[core].values, tr["ret_12m_fwd"].values)
        yp = m.predict(te[core].values); yt = te["ret_12m_fwd"].values
        sp = spearmanr(yt,yp).correlation if len(set(yt))>1 else np.nan
        metrics.append({"year":ty,"rmse":np.sqrt(mean_squared_error(yt,yp)),"mae":mean_absolute_error(yt,yp),"r2":r2_score(yt,yp),"spearman_ic":sp,"n":len(te)})
        fis.append(pd.Series(m.feature_importances_,index=core,name=f"y{ty}"))
    metrics_df = pd.DataFrame(metrics)
    fi_df = pd.concat(fis,axis=1).T if fis else pd.DataFrame()
    latest = me.dropna(subset=core).sort_values("data").groupby("ticker").tail(1).copy()
    if len(data_ml)>=20:
        train_all = data_ml.dropna(subset=core+["ret_12m_fwd"])
        mfin = RandomForestRegressor(n_estimators=400,max_depth=6,min_samples_leaf=3,random_state=42,n_jobs=-1)
        mfin.fit(train_all[core].values, train_all["ret_12m_fwd"].values)
        latest["pred_ret_12m"] = mfin.predict(latest[core].values)
        latest["rank_pred"] = latest["pred_ret_12m"].rank(ascending=False,method="min").astype(int)
    else:
        latest["pred_ret_12m"] = np.nan; latest["rank_pred"] = np.nan
    clust = latest.dropna(subset=mkt).copy()
    if len(clust)>=3:
        sc = StandardScaler(); Xs = sc.fit_transform(clust[mkt].values)
        km = KMeans(n_clusters=min(3,len(clust)),random_state=42,n_init=20)
        clust["cluster"] = km.fit_predict(Xs)
        centers = pd.DataFrame(sc.inverse_transform(km.cluster_centers_),columns=mkt)
        names = {}; used = set()
        q = centers["vol_21"].idxmin(); names[q]="Defensivo"; used.add(q)
        remaining = [i for i in centers.index if i not in used]
        if remaining:
            g2 = centers.loc[remaining,"mom_6m"].idxmax(); names[g2]="Crescimento"; used.add(g2)
        for i in centers.index:
            if i not in used: names[i]="Risco"
        clust["perfil"] = clust["cluster"].map(names)
    else:
        clust["cluster"]=0; clust["perfil"]="N/A"
    return metrics_df, fi_df, latest, clust

# === TOOL 4: MODELO PREDITIVO ===
def run_predictive_model(prices):
    results, all_fi = [], []
    for ticker in prices["ticker"].unique():
        sub = prices[prices["ticker"]==ticker][["data","preco"]].copy().sort_values("data").set_index("data")
        if len(sub)<100: continue
        df = pd.DataFrame()
        df["preco"] = sub["preco"]
        df["retorno"] = df["preco"].pct_change()
        df["retorno_1d"] = df["retorno"].shift(1)
        df["retorno_5d"] = df["preco"].pct_change(5)
        df["retorno_20d"] = df["preco"].pct_change(20)
        df["vol_20d"] = df["retorno"].rolling(20).std()
        df["acima_media20"] = np.where(df["preco"]>df["preco"].rolling(20).mean(),1,0)
        df["target"] = np.where(df["retorno"].shift(-1)>0,1,0)
        df = df.dropna()
        if len(df)<50: continue
        feats = ["retorno_1d","retorno_5d","retorno_20d","vol_20d","acima_media20"]
        X, y = df[feats], df["target"]
        split = int(len(df)*0.7)
        m = RandomForestClassifier(n_estimators=300,max_depth=4,random_state=42,n_jobs=-1)
        m.fit(X.iloc[:split], y.iloc[:split])
        y_pred = m.predict(X.iloc[split:]); y_prob = m.predict_proba(X.iloc[split:])[:,1]
        acc = accuracy_score(y.iloc[split:], y_pred)
        auc = roc_auc_score(y.iloc[split:], y_prob) if len(set(y.iloc[split:]))>1 else np.nan
        results.append({"ticker":ticker,"accuracy":round(acc,4),"auc":round(auc,4),"n_test":len(y)-split})
        all_fi.append(pd.Series(m.feature_importances_,index=feats,name=ticker))
    return pd.DataFrame(results), (pd.DataFrame(all_fi) if all_fi else pd.DataFrame())

# === TOOL 5: TRADING SYSTEM ===
def run_trading_system(prices, mc=20, ml=60):
    all_res, summary = [], []
    for ticker in prices["ticker"].unique():
        sub = prices[prices["ticker"]==ticker][["data","preco"]].copy().sort_values("data")
        if len(sub)<ml+10: continue
        df = pd.DataFrame({"data":sub["data"].values,"preco":sub["preco"].values})
        df["ma_c"] = df["preco"].rolling(mc).mean()
        df["ma_l"] = df["preco"].rolling(ml).mean()
        df["sinal"] = np.where(df["ma_c"]>df["ma_l"],1,0)
        df["ret_acao"] = df["preco"].pct_change()
        df["ret_est"] = df["sinal"].shift(1)*df["ret_acao"]
        df = df.dropna()
        df["acum_acao"] = (1+df["ret_acao"]).cumprod()
        df["acum_est"] = (1+df["ret_est"]).cumprod()
        df["ticker"] = ticker
        all_res.append(df)
        rb = df["acum_acao"].iloc[-1]-1; rs = df["acum_est"].iloc[-1]-1
        summary.append({"ticker":ticker,"ret_buyhold":round(rb*100,1),"ret_estrategia":round(rs*100,1),"alpha":round((rs-rb)*100,1)})
    return (pd.concat(all_res,ignore_index=True) if all_res else pd.DataFrame()), pd.DataFrame(summary)

# === TOOL 6: MONTE CARLO ===
def run_monte_carlo(prices, n_sim=10000):
    wide = prices.pivot_table(index="data",columns="ticker",values="preco")
    wide = wide.dropna(axis=1,how="all").dropna()
    tickers_ok = wide.columns.tolist()
    if len(tickers_ok)<2: return pd.DataFrame(), {}, tickers_ok
    rets = wide.pct_change().dropna()
    media, cov = rets.mean(), rets.cov()
    results = []
    for _ in range(n_sim):
        w = np.random.random(len(tickers_ok)); w = w/w.sum()
        ret_a = np.dot(w,media)*252
        risk_a = np.sqrt(np.dot(w.T,np.dot(cov*252,w)))
        sharpe = ret_a/risk_a if risk_a>0 else 0
        results.append(list(w)+[ret_a,risk_a,sharpe])
    cols = [f"w_{t}" for t in tickers_ok]+["retorno","risco","sharpe"]
    df = pd.DataFrame(results,columns=cols)
    best = df.loc[df["sharpe"].idxmax()]
    best_dict = {"sharpe":best["sharpe"],"retorno":best["retorno"],"risco":best["risco"]}
    for t in tickers_ok: best_dict[t] = best[f"w_{t}"]
    return df, best_dict, tickers_ok

# === TOOL 7: SCORING ===
PERFIS = {
    "conservador": {"mercado":0.60,"qualidade":0.40},
    "moderado":    {"mercado":0.50,"qualidade":0.50},
    "agressivo":   {"mercado":0.40,"qualidade":0.60},
}

def run_scoring(prices, fund_df, perfil="moderado"):
    pesos = PERFIS.get(perfil, PERFIS["moderado"])
    snap = prices.dropna(subset=["vol_21","mom_6m"]).sort_values("data").groupby("ticker").tail(1).copy()
    fund_cols = [c for c in ["ticker","P_L","P_VP","EV_EBITDA","nome","setor","marketCap","div_yield","profitMargins","returnOnEquity","revenueGrowth"] if c in fund_df.columns]
    snap = snap.merge(fund_df[fund_cols], on="ticker", how="left")
    snap["sc_mom"] = snap["mom_6m"].rank(pct=True)*100
    snap["sc_vol"] = snap["vol_21"].rank(pct=True,ascending=False)*100
    snap["sc_dd"] = snap["drawdown"].rank(pct=True,ascending=False)*100
    snap["score_mercado"] = snap["sc_mom"]*0.5+snap["sc_vol"]*0.3+snap["sc_dd"]*0.2
    if "P_VP" in snap.columns and snap["P_VP"].notna().sum()>0:
        snap["score_qual"] = snap["P_VP"].rank(pct=True,ascending=True)*100
    else:
        snap["score_qual"] = 50.0
    snap["score_qual"] = snap["score_qual"].fillna(50)
    snap["score"] = (snap["score_mercado"]*pesos["mercado"]+snap["score_qual"]*pesos["qualidade"]).round(1)
    def rec(s):
        if pd.isna(s): return "HOLD"
        if s>=65: return "BUY"
        if s>=35: return "HOLD"
        return "SELL"
    snap["recomendacao"] = snap["score"].apply(rec)
    snap["rank"] = snap["score"].rank(ascending=False,method="min").astype(int)
    for col in ["nome","setor","P_L","P_VP","EV_EBITDA","marketCap","div_yield"]:
        if col not in snap.columns:
            snap[col] = np.nan if col not in ["nome","setor"] else "-"
    return snap.sort_values("rank")
