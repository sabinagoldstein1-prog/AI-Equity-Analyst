"""
engine.py — Motor do Agente de IA
6 tools orquestradas pelo agente principal.
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
np.random.seed(42)

def safe_div(a, b):
    try:
        a, b = float(a), float(b)
        if np.isnan(a) or np.isnan(b) or b == 0: return np.nan
        return a / b
    except: return np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — PREÇOS E MÉTRICAS DE MERCADO
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_prices(tickers, start="2021-01-01"):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series): raw = raw.to_frame(name=tickers[0])
    p = raw.stack(future_stack=True).reset_index()
    p.columns = ["data", "ticker", "preco"]
    p["data"] = pd.to_datetime(p["data"])
    p = p.dropna(subset=["preco"]).sort_values(["ticker", "data"])
    g = p.groupby("ticker")
    p["ret_dia"]  = g["preco"].pct_change()
    p["vol_21"]   = g["ret_dia"].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    p["mom_6m"]   = g["preco"].transform(lambda x: x.pct_change(126))
    p["mom_12m"]  = g["preco"].transform(lambda x: x.pct_change(252))
    p["drawdown"] = g["preco"].transform(lambda x: (x - x.cummax()) / x.cummax())
    return p

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — FUNDAMENTOS E MÚLTIPLOS (Yahoo Finance)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_fundamentals(tickers):
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mcap   = info.get("marketCap") or np.nan
            shares = info.get("sharesOutstanding") or info.get("floatShares") or np.nan
            pl_r   = info.get("trailingPE") or np.nan
            pvp_r  = info.get("priceToBook") or np.nan
            ev_eb  = info.get("enterpriseToEbitda") or np.nan
            ev     = info.get("enterpriseValue") or np.nan
            debt   = info.get("totalDebt") or np.nan
            price  = info.get("currentPrice") or info.get("previousClose") or np.nan
            name   = info.get("shortName", t)
            sector = info.get("sector", "?")
            summary = (info.get("longBusinessSummary", "") or "").lower()
            dy     = info.get("dividendYield") or np.nan
            lucro  = safe_div(mcap, pl_r)
            pl_eq  = safe_div(mcap, pvp_r)
            if pd.isna(ev_eb) and pd.notna(ev) and pd.notna(lucro) and lucro > 0:
                ev_eb = safe_div(ev, lucro)
            rows.append({"ticker": t, "nome": name, "setor": sector,
                         "preco": price, "marketCap": mcap, "shares": shares,
                         "P_L": pl_r, "P_VP": pvp_r, "EV_EBITDA": ev_eb,
                         "EV": ev, "lucro": lucro, "pl_equity": pl_eq,
                         "totalDebt": debt, "summary": summary, "div_yield": dy})
        except:
            rows.append({"ticker": t, "nome": t, "setor": "?"})
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — NLP: SENTIMENTO E TÓPICOS DE RISCO
# ═══════════════════════════════════════════════════════════════════════════════
_POS = ["growth","strong","solid","increase","improved","positive","stable","efficient",
        "profitable","record","expansion","robust","resilient","diversified","leading",
        "innovation","sustainable","premium","competitive","opportunity"]
_NEG = ["risk","decline","loss","debt","volatile","uncertain","litigation","restructuring",
        "impairment","default","downgrade","challenge","concern","investigation","fraud",
        "weakness","deficit","contingency","sanction","penalty","crisis","instability"]
_TOPICS = {
    "alavancagem": ["debt","leverage","borrowing","loan","financing"],
    "litigio":     ["litigation","lawsuit","legal","penalty","sanction"],
    "governanca":  ["governance","compliance","transparency","audit","ethics"],
    "regulatorio": ["regulatory","regulation","government","tariff","concession"],
}

def run_nlp(fund_df):
    rows = []
    for _, r in fund_df.iterrows():
        text = r.get("summary", "") or ""
        pos = sum(text.count(w) for w in _POS)
        neg = sum(text.count(w) for w in _NEG)
        total = pos + neg
        score = 0.0 if total == 0 else (pos - neg) / total
        label = "positivo" if score > 0.1 else "negativo" if score < -0.1 else "neutro"
        topics = {f"topic_{k}": sum(text.count(w) for w in v) for k, v in _TOPICS.items()}
        rows.append({"ticker": r["ticker"], "sent_score": round(score, 3),
                     "sent_label": label, "n_pos": pos, "n_neg": neg, **topics})
    df = pd.DataFrame(rows)
    mn, mx = df["sent_score"].min(), df["sent_score"].max()
    rng = mx - mn if mx - mn > 1e-9 else 1
    df["indice_textual"] = ((df["sent_score"] - mn) / rng * 100).round(1)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — VALUATION + CENÁRIOS MACRO
# ═══════════════════════════════════════════════════════════════════════════════
_SELIC = -0.05
_EXPORTERS = ["PETR4.SA","PRIO3.SA","VALE3.SA","SUZB3.SA","KLBN11.SA"]
_SCENARIOS = {
    "dove -200bp": {"selic_bp": -200, "fx_pct": 0.0},
    "base":        {"selic_bp": 0,    "fx_pct": 0.0},
    "hawk +200bp": {"selic_bp": 200,  "fx_pct": 0.0},
    "fx up +10%":  {"selic_bp": 0,    "fx_pct": 0.10},
    "fx dn -10%":  {"selic_bp": 0,    "fx_pct": -0.10},
}

def run_valuation(fund_df):
    df = fund_df.copy()
    for c in ["P_L","P_VP","EV_EBITDA"]:
        df[f"{c}_ok"] = df[c].where((df[c] > 0) & (df[c] <= 200))
    med = df.groupby("setor")[[f"{c}_ok" for c in ["P_L","P_VP","EV_EBITDA"]]].median()
    med.columns = ["P_L_med","P_VP_med","EV_EBITDA_med"]
    glob = df[[f"{c}_ok" for c in ["P_L","P_VP","EV_EBITDA"]]].median()
    med = med.fillna(dict(zip(med.columns, glob.values)))
    df = df.merge(med, left_on="setor", right_index=True, how="left")
    df["impl_PL"]  = df.apply(lambda r: safe_div(r["lucro"]*r["P_L_med"], r["shares"]), axis=1)
    df["impl_PVP"] = df.apply(lambda r: safe_div(r["pl_equity"]*r["P_VP_med"], r["shares"]), axis=1)
    def _ev(r):
        ebitda = safe_div(r["EV"], r["EV_EBITDA"])
        if pd.isna(ebitda): return np.nan
        return safe_div(ebitda*r["EV_EBITDA_med"]-(r["totalDebt"] or 0), r["shares"])
    df["impl_EV"] = df.apply(_ev, axis=1)
    df["impl_price"] = df[["impl_PL","impl_PVP","impl_EV"]].apply(
        lambda row: np.nanmedian([v for v in row if pd.notna(v) and v>0])
        if any(pd.notna(v) and v>0 for v in row) else np.nan, axis=1)
    df["gap_pct"] = (df["impl_price"]/df["preco"]-1)*100
    def _vs(ratio):
        if pd.isna(ratio): return 50.0
        return max(0, min(100, (float(ratio)-0.75)/0.50*100))
    df["val_score"] = (df["impl_price"]/df["preco"]).apply(_vs)
    sens = []
    for scen, p in _SCENARIOS.items():
        sf = 1.0+_SELIC*(p["selic_bp"]/100.0)
        for _, r in df.iterrows():
            fx = (1.0+p["fx_pct"]) if r["ticker"] in _EXPORTERS else 1.0
            la = (r["lucro"] or 0)*fx
            impls = [safe_div(la*(r["P_L_med"] or 0)*sf, r["shares"]),
                     safe_div((r["pl_equity"] or 0)*(r["P_VP_med"] or 0)*sf, r["shares"])]
            valid = [v for v in impls if pd.notna(v) and v>0]
            impl_s = float(np.median(valid)) if valid else np.nan
            gap = (impl_s/r["preco"]-1)*100 if pd.notna(impl_s) and r["preco"]>0 else np.nan
            sens.append({"scenario": scen, "ticker": r["ticker"], "gap_pct": gap})
    return df, pd.DataFrame(sens)

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — RANDOM FOREST WALK-FORWARD + CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
def run_ml(prices):
    feat = prices.copy()
    feat["month"] = feat["data"].dt.to_period("M")
    me = feat.sort_values("data").groupby(["ticker","month"]).tail(1).copy()
    me = me.sort_values(["ticker","data"])
    mkt = ["vol_21","mom_6m","mom_12m","drawdown"]
    for c in mkt:
        me[f"{c}_z"] = me.groupby("data")[c].transform(
            lambda s: (s-s.mean())/(s.std() if s.std()>0 else 1))
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
        m = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
        m.fit(tr[core].values, tr["ret_12m_fwd"].values)
        yp = m.predict(te[core].values); yt = te["ret_12m_fwd"].values
        sp = spearmanr(yt, yp).correlation if len(set(yt))>1 else np.nan
        metrics.append({"year":ty, "rmse":np.sqrt(mean_squared_error(yt,yp)),
                        "mae":mean_absolute_error(yt,yp), "r2":r2_score(yt,yp),
                        "spearman_ic":sp, "n":len(te)})
        fis.append(pd.Series(m.feature_importances_, index=core, name=f"y{ty}"))
    metrics_df = pd.DataFrame(metrics)
    fi_df = pd.concat(fis, axis=1).T if fis else pd.DataFrame()
    # Previsões finais
    latest = me.dropna(subset=core).sort_values("data").groupby("ticker").tail(1).copy()
    if len(data_ml) >= 20:
        train_all = data_ml.dropna(subset=core+["ret_12m_fwd"])
        mfin = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
        mfin.fit(train_all[core].values, train_all["ret_12m_fwd"].values)
        latest["pred_ret_12m"] = mfin.predict(latest[core].values)
        latest["rank_pred"] = latest["pred_ret_12m"].rank(ascending=False, method="min").astype(int)
    else:
        latest["pred_ret_12m"] = np.nan; latest["rank_pred"] = np.nan
    # Clustering
    clust = latest.dropna(subset=mkt).copy()
    if len(clust) >= 3:
        sc = StandardScaler(); Xs = sc.fit_transform(clust[mkt].values)
        km = KMeans(n_clusters=min(3,len(clust)), random_state=42, n_init=20)
        clust["cluster"] = km.fit_predict(Xs)
        centers = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=mkt)
        names = {}; used = set()
        q = centers["vol_21"].idxmin(); names[q]="🛡️ Defensivo"; used.add(q)
        remaining = [i for i in centers.index if i not in used]
        if remaining:
            g = centers.loc[remaining,"mom_6m"].idxmax(); names[g]="🚀 Crescimento"; used.add(g)
        for i in centers.index:
            if i not in used: names[i]="⚡ Risco"
        clust["perfil"] = clust["cluster"].map(names)
    else:
        clust["cluster"] = 0; clust["perfil"] = "N/A"
    return metrics_df, fi_df, latest, clust

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — SCORING COMPOSTO + RECOMENDAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
PERFIS = {
    "conservador": {"mercado":0.50,"valuation":0.15,"nlp":0.15,"qualidade":0.20},
    "moderado":    {"mercado":0.35,"valuation":0.25,"nlp":0.15,"qualidade":0.25},
    "agressivo":   {"mercado":0.25,"valuation":0.35,"nlp":0.10,"qualidade":0.30},
}

def run_scoring(prices, fund_df, nlp_df, val_df, ml_preds, perfil="moderado"):
    pesos = PERFIS.get(perfil, PERFIS["moderado"])
    snap = prices.dropna(subset=["vol_21","mom_6m"]).sort_values("data").groupby("ticker").tail(1).copy()
    # Safe merges: only use columns that exist in source DataFrames
    fund_cols = [c for c in ["ticker","P_L","P_VP","EV_EBITDA","nome","setor","marketCap","div_yield"] if c in fund_df.columns]
    snap = snap.merge(fund_df[fund_cols], on="ticker", how="left")
    nlp_cols = [c for c in ["ticker","indice_textual","sent_label","sent_score"] if c in nlp_df.columns]
    snap = snap.merge(nlp_df[nlp_cols], on="ticker", how="left")
    val_cols = [c for c in ["ticker","val_score","impl_price","gap_pct"] if c in val_df.columns]
    snap = snap.merge(val_df[val_cols], on="ticker", how="left")
    if ml_preds is not None and not ml_preds.empty:
        ml_cols = [c for c in ["ticker","pred_ret_12m","rank_pred"] if c in ml_preds.columns]
        snap = snap.merge(ml_preds[ml_cols], on="ticker", how="left")
    # Fill NaN with neutral values
    for col in ["indice_textual","val_score"]:
        if col in snap.columns:
            snap[col] = snap[col].fillna(50)
        else:
            snap[col] = 50
    # Ensure P_VP exists for quality score
    if "P_VP" not in snap.columns:
        snap["P_VP"] = np.nan
    snap["sc_mom"] = snap["mom_6m"].rank(pct=True)*100
    snap["sc_vol"] = snap["vol_21"].rank(pct=True, ascending=False)*100
    snap["sc_dd"]  = snap["drawdown"].rank(pct=True, ascending=False)*100
    snap["score_mercado"] = snap["sc_mom"]*0.5+snap["sc_vol"]*0.3+snap["sc_dd"]*0.2
    # Quality score: if P_VP is all NaN, use neutral 50
    if snap["P_VP"].notna().sum() > 0:
        snap["score_qual"] = snap["P_VP"].rank(pct=True, ascending=True)*100
    else:
        snap["score_qual"] = 50
    snap["score_qual"] = snap["score_qual"].fillna(50)
    snap["score"] = (
        snap["score_mercado"]*pesos["mercado"]+
        snap["val_score"]*pesos["valuation"]+
        snap["indice_textual"]*pesos["nlp"]+
        snap["score_qual"]*pesos["qualidade"]
    ).round(1)
    def rec(s):
        if pd.isna(s): return "🟡 Hold"
        if s >= 65: return "🟢 Buy"
        if s >= 35: return "🟡 Hold"
        return "🔴 Sell"
    snap["recomendacao"] = snap["score"].apply(rec)
    snap["rank"] = snap["score"].rank(ascending=False, method="min").astype(int)
    # Ensure all display columns exist
    for col in ["nome","setor","P_L","P_VP","EV_EBITDA","marketCap","gap_pct","sent_label","impl_price"]:
        if col not in snap.columns:
            snap[col] = np.nan if col != "nome" and col != "setor" and col != "sent_label" else "?"
    return snap.sort_values("rank")

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — MODELO PREDITIVO: Prever se ação sobe amanhã (Aula FinIA - Seção 1)
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def run_predictive_model(prices):
    """
    Para cada ticker, treina um modelo preditivo que prevê se a ação sobe amanhã.
    Usa: retorno 1d, 5d, 20d, volatilidade 20d, posição vs média 20d.
    Split temporal 70/30. Retorna métricas e importância das features.
    """
    results = []
    all_fi = []
    for ticker in prices["ticker"].unique():
        sub = prices[prices["ticker"] == ticker][["data", "preco"]].copy()
        sub = sub.sort_values("data").set_index("data")
        if len(sub) < 100:
            continue
        df = pd.DataFrame()
        df["preco"] = sub["preco"]
        df["retorno"] = df["preco"].pct_change()
        df["retorno_1d"] = df["retorno"].shift(1)
        df["retorno_5d"] = df["preco"].pct_change(5)
        df["retorno_20d"] = df["preco"].pct_change(20)
        df["vol_20d"] = df["retorno"].rolling(20).std()
        df["media_20"] = df["preco"].rolling(20).mean()
        df["acima_media20"] = np.where(df["preco"] > df["media_20"], 1, 0)
        df["target"] = np.where(df["retorno"].shift(-1) > 0, 1, 0)
        df = df.dropna()
        if len(df) < 50:
            continue
        feats = ["retorno_1d", "retorno_5d", "retorno_20d", "vol_20d", "acima_media20"]
        X = df[feats]
        y = df["target"]
        split = int(len(df) * 0.7)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]
        m = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_prob = m.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob) if len(set(y_te)) > 1 else np.nan
        results.append({"ticker": ticker, "accuracy": round(acc, 4), "auc": round(auc, 4), "n_test": len(y_te)})
        all_fi.append(pd.Series(m.feature_importances_, index=feats, name=ticker))
    metrics = pd.DataFrame(results)
    fi = pd.DataFrame(all_fi) if all_fi else pd.DataFrame()
    return metrics, fi

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 8 — TRADING SYSTEM: Médias Móveis (Aula FinIA - Seção 2)
# ═══════════════════════════════════════════════════════════════════════════════
def run_trading_system(prices, media_curta=20, media_longa=60):
    """
    Estratégia de cruzamento de médias móveis para cada ticker.
    Retorna DataFrame com retorno acumulado da estratégia vs buy-and-hold.
    """
    all_results = []
    for ticker in prices["ticker"].unique():
        sub = prices[prices["ticker"] == ticker][["data", "preco"]].copy().sort_values("data")
        if len(sub) < media_longa + 10:
            continue
        df = pd.DataFrame()
        df["data"] = sub["data"].values
        df["preco"] = sub["preco"].values
        df["ma_curta"] = df["preco"].rolling(media_curta).mean()
        df["ma_longa"] = df["preco"].rolling(media_longa).mean()
        df["sinal"] = np.where(df["ma_curta"] > df["ma_longa"], 1, 0)
        df["ret_acao"] = df["preco"].pct_change()
        df["ret_estrategia"] = df["sinal"].shift(1) * df["ret_acao"]
        df = df.dropna()
        df["acum_acao"] = (1 + df["ret_acao"]).cumprod()
        df["acum_estrategia"] = (1 + df["ret_estrategia"]).cumprod()
        df["ticker"] = ticker
        all_results.append(df)
    if not all_results:
        return pd.DataFrame()
    combined = pd.concat(all_results, ignore_index=True)
    # Resumo final por ticker
    summary = []
    for ticker in combined["ticker"].unique():
        sub = combined[combined["ticker"] == ticker]
        ret_bh = sub["acum_acao"].iloc[-1] - 1
        ret_st = sub["acum_estrategia"].iloc[-1] - 1
        summary.append({"ticker": ticker, "ret_buyhold": round(ret_bh * 100, 1),
                        "ret_estrategia": round(ret_st * 100, 1),
                        "alpha": round((ret_st - ret_bh) * 100, 1)})
    return combined, pd.DataFrame(summary)

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 9 — MONTE CARLO: Otimização de Portfólio (Aula FinIA - Seção 4)
# ═══════════════════════════════════════════════════════════════════════════════
def run_monte_carlo(prices, n_sim=10000):
    """
    Simula n_sim carteiras com pesos aleatórios.
    Retorna: (df_carteiras, melhor_carteira, tickers_usados)
    """
    # Monta tabela de retornos diários (wide)
    wide = prices.pivot_table(index="data", columns="ticker", values="preco")
    wide = wide.dropna(axis=1, how="all").dropna()
    tickers_ok = wide.columns.tolist()
    if len(tickers_ok) < 2:
        return pd.DataFrame(), {}, tickers_ok
    rets = wide.pct_change().dropna()
    media = rets.mean()
    cov = rets.cov()
    results = []
    for _ in range(n_sim):
        w = np.random.random(len(tickers_ok))
        w = w / w.sum()
        ret_anual = np.dot(w, media) * 252
        risk_anual = np.sqrt(np.dot(w.T, np.dot(cov * 252, w)))
        sharpe = ret_anual / risk_anual if risk_anual > 0 else 0
        results.append(list(w) + [ret_anual, risk_anual, sharpe])
    cols = [f"w_{t}" for t in tickers_ok] + ["retorno", "risco", "sharpe"]
    df = pd.DataFrame(results, columns=cols)
    best = df.loc[df["sharpe"].idxmax()]
    # Formata melhor carteira
    best_dict = {"sharpe": best["sharpe"], "retorno": best["retorno"], "risco": best["risco"]}
    for t in tickers_ok:
        best_dict[t] = best[f"w_{t}"]
    return df, best_dict, tickers_ok
