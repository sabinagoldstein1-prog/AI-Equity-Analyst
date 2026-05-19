"""
engine.py - AI Equity Analyst Engine v5
Multiple fallback strategies for robust Yahoo Finance data
"""
import warnings
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
np.random.seed(42)


# ============================================================
# UTILITIES
# ============================================================

def safe_div(a, b):
    try:
        a, b = float(a), float(b)
        if np.isnan(a) or np.isnan(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def to_float(v):
    """Convert anything to float, returning NaN on failure."""
    if v is None:
        return np.nan
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return np.nan
        return f
    except (TypeError, ValueError):
        return np.nan


def to_str(v, default="-"):
    """Convert anything to clean string."""
    if v is None:
        return default
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan", "?", "n/a", "null"):
        return default
    return s


# ============================================================
# TOOL 1: PRICES & MARKET METRICS
# ============================================================

def fetch_prices(tickers, start="2021-01-01"):
    """Download adjusted close prices and compute market metrics."""
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    p = raw.stack(future_stack=True).reset_index()
    p.columns = ["data", "ticker", "preco"]
    p["data"] = pd.to_datetime(p["data"])
    p = p.dropna(subset=["preco"]).sort_values(["ticker", "data"])
    g = p.groupby("ticker")
    p["ret_dia"] = g["preco"].pct_change()
    p["vol_21"] = g["ret_dia"].transform(
        lambda x: x.rolling(21).std() * np.sqrt(252))
    p["mom_6m"] = g["preco"].transform(lambda x: x.pct_change(126))
    p["mom_12m"] = g["preco"].transform(lambda x: x.pct_change(252))
    p["drawdown"] = g["preco"].transform(
        lambda x: (x - x.cummax()) / x.cummax())
    return p


# ============================================================
# TOOL 2: FUNDAMENTALS (Robust 3-layer fetching)
# ============================================================

# Sector/Name maps from b3_universe (single source of truth)
try:
    from b3_universe import B3_UNIVERSE
    SECTOR_MAP = {t: s for t, (_, s) in B3_UNIVERSE.items()}
    NAME_MAP = {t: n for t, (n, _) in B3_UNIVERSE.items()}
except ImportError:
    # Fallback if b3_universe not available
    SECTOR_MAP = {}
    NAME_MAP = {}


def _empty_row(ticker):
    """Default row with NaN/dash for all fields."""
    return {
        "ticker": ticker,
        "nome": NAME_MAP.get(ticker, ticker.replace(".SA", "")),
        "setor": SECTOR_MAP.get(ticker, "-"),
        "preco": np.nan, "marketCap": np.nan, "shares": np.nan,
        "P_L": np.nan, "P_VP": np.nan, "EV_EBITDA": np.nan,
        "EV": np.nan, "lucro": np.nan, "pl_equity": np.nan,
        "totalDebt": np.nan, "div_yield": np.nan,
        "profitMargins": np.nan, "returnOnEquity": np.nan,
        "revenueGrowth": np.nan, "ebitdaMargins": np.nan,
        "debtToEquity": np.nan,
    }


def fetch_fundamentals(tickers, prices_df=None):
    """
    Fetch fundamentals with 3 fallback layers:
    1. yf.Ticker().info (primary)
    2. yf.Ticker().fast_info (price/mcap/shares)
    3. Last price from prices_df + hardcoded name/sector maps
    """
    rows = []
    for idx, t in enumerate(tickers):
        r = _empty_row(t)

        # ===== Layer 1: .info =====
        info = {}
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            pass

        if info:
            # Name & sector (with fallback to hardcoded maps)
            yf_name = info.get("shortName") or info.get("longName")
            if yf_name:
                r["nome"] = to_str(yf_name, r["nome"])
            yf_sector = info.get("sector") or info.get("industry")
            if yf_sector:
                r["setor"] = to_str(yf_sector, r["setor"])

            # Price
            r["preco"] = to_float(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )

            # Market data
            r["marketCap"] = to_float(info.get("marketCap"))
            r["shares"] = to_float(
                info.get("sharesOutstanding") or info.get("floatShares"))

            # Multiples
            r["P_L"] = to_float(
                info.get("trailingPE") or info.get("forwardPE"))
            r["P_VP"] = to_float(info.get("priceToBook"))
            r["EV_EBITDA"] = to_float(info.get("enterpriseToEbitda"))
            r["EV"] = to_float(info.get("enterpriseValue"))
            r["totalDebt"] = to_float(info.get("totalDebt"))
            r["div_yield"] = to_float(info.get("dividendYield"))
            r["profitMargins"] = to_float(info.get("profitMargins"))
            r["returnOnEquity"] = to_float(info.get("returnOnEquity"))
            r["revenueGrowth"] = to_float(info.get("revenueGrowth"))
            r["ebitdaMargins"] = to_float(info.get("ebitdaMargins"))
            r["debtToEquity"] = to_float(info.get("debtToEquity"))

        # ===== Layer 2: fast_info (if price still missing) =====
        if pd.isna(r["preco"]) or pd.isna(r["marketCap"]):
            try:
                tk = yf.Ticker(t)
                fi = tk.fast_info
                if pd.isna(r["preco"]):
                    r["preco"] = to_float(
                        getattr(fi, "last_price", None)
                        or getattr(fi, "previous_close", None))
                if pd.isna(r["marketCap"]):
                    r["marketCap"] = to_float(getattr(fi, "market_cap", None))
                if pd.isna(r["shares"]):
                    r["shares"] = to_float(getattr(fi, "shares", None))
            except Exception:
                pass

        # ===== Layer 3: prices_df fallback for price =====
        if pd.isna(r["preco"]) and prices_df is not None:
            try:
                sub = prices_df[prices_df["ticker"] == t]
                if not sub.empty:
                    r["preco"] = to_float(sub["preco"].iloc[-1])
            except Exception:
                pass

        # Derived
        r["lucro"] = safe_div(r["marketCap"], r["P_L"])
        r["pl_equity"] = safe_div(r["marketCap"], r["P_VP"])

        # EV/EBITDA fallback for banks (use EV/Lucro)
        if pd.isna(r["EV_EBITDA"]) and pd.notna(r["EV"]) and pd.notna(r["lucro"]) and r["lucro"] > 0:
            r["EV_EBITDA"] = safe_div(r["EV"], r["lucro"])

        rows.append(r)

        # Rate limit protection
        if idx < len(tickers) - 1:
            time.sleep(0.2)

    return pd.DataFrame(rows)


# ============================================================
# TOOL 3: ML WALK-FORWARD + CLUSTERING
# ============================================================

def run_ml(prices, fund_df=None):
    """
    Random Forest walk-forward for 12m return prediction + KMeans clustering.

    If fund_df is provided, merges fundamental features (P/L, P/VP, ROE, margin, growth)
    so the model can learn relationships beyond pure technicals. This is critical
    for stocks with strong fundamentals but weak momentum (e.g. quality banks).
    """
    feat = prices.copy()
    feat["month"] = feat["data"].dt.to_period("M")
    me = feat.sort_values("data").groupby(["ticker", "month"]).tail(1).copy()
    me = me.sort_values(["ticker", "data"])

    # Market features
    mkt = ["vol_21", "mom_6m", "mom_12m", "drawdown"]
    for c in mkt:
        me[f"{c}_z"] = me.groupby("data")[c].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1))
    mkt_z = [f"{c}_z" for c in mkt]

    # Fundamental features (merged from fund_df if provided)
    fund_cols = []
    if fund_df is not None and not fund_df.empty:
        fund_feats = ["P_L", "P_VP", "returnOnEquity", "profitMargins",
                      "revenueGrowth", "ebitdaMargins", "debtToEquity"]
        fund_avail = [c for c in fund_feats if c in fund_df.columns]
        if fund_avail:
            me = me.merge(fund_df[["ticker"] + fund_avail], on="ticker", how="left")
            # Cross-sectional z-scores for fundamentals (per date)
            for c in fund_avail:
                me[f"{c}_z"] = me.groupby("data")[c].transform(
                    lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1))
                # Fill NaN with 0 (sector neutral)
                me[f"{c}_z"] = me[f"{c}_z"].fillna(0)
            fund_cols = [f"{c}_z" for c in fund_avail]

    me["ret_12m_fwd"] = me.groupby("ticker")["preco"].transform(
        lambda s: s.shift(-12) / s - 1.0)

    # Combined feature set: market + fundamental
    core = mkt + mkt_z + fund_cols
    data_ml = me.dropna(subset=mkt + mkt_z + ["ret_12m_fwd"]).copy()
    if fund_cols:
        # Don't drop on fund_cols (we filled with 0), but ensure they exist
        for c in fund_cols:
            if c not in data_ml.columns:
                data_ml[c] = 0
    data_ml["year"] = data_ml["data"].dt.year

    yrs = sorted(data_ml["year"].unique())
    metrics, fis = [], []
    for ty in yrs[2:]:
        tr = data_ml[data_ml["year"] < ty]
        te = data_ml[data_ml["year"] == ty]
        if len(tr) < 20 or len(te) < 3:
            continue
        m = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1)
        m.fit(tr[core].values, tr["ret_12m_fwd"].values)
        yp = m.predict(te[core].values)
        yt = te["ret_12m_fwd"].values
        sp = spearmanr(yt, yp).correlation if len(set(yt)) > 1 else np.nan
        metrics.append({
            "year": ty,
            "rmse": np.sqrt(mean_squared_error(yt, yp)),
            "mae": mean_absolute_error(yt, yp),
            "r2": r2_score(yt, yp),
            "spearman_ic": sp, "n": len(te),
        })
        fis.append(pd.Series(m.feature_importances_, index=core, name=f"y{ty}"))

    metrics_df = pd.DataFrame(metrics)
    fi_df = pd.concat(fis, axis=1).T if fis else pd.DataFrame()

    # Final model for predictions
    # For final prediction, only require market features (fundamentals already filled with 0)
    latest = me.dropna(subset=mkt + mkt_z).sort_values("data").groupby("ticker").tail(1).copy()
    # Ensure fund cols exist (fill with 0 if missing)
    for c in fund_cols:
        if c not in latest.columns:
            latest[c] = 0
        else:
            latest[c] = latest[c].fillna(0)
    if len(data_ml) >= 20:
        train_all = data_ml.dropna(subset=mkt + mkt_z + ["ret_12m_fwd"])
        mfin = RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1)
        mfin.fit(train_all[core].values, train_all["ret_12m_fwd"].values)
        latest["pred_ret_12m"] = mfin.predict(latest[core].values)
        latest["rank_pred"] = latest["pred_ret_12m"].rank(
            ascending=False, method="min").astype(int)
    else:
        latest["pred_ret_12m"] = np.nan
        latest["rank_pred"] = np.nan

    # Clustering
    clust = latest.dropna(subset=mkt).copy()
    if len(clust) >= 3:
        sc = StandardScaler()
        Xs = sc.fit_transform(clust[mkt].values)
        km = KMeans(n_clusters=min(3, len(clust)), random_state=42, n_init=20)
        clust["cluster"] = km.fit_predict(Xs)
        centers = pd.DataFrame(
            sc.inverse_transform(km.cluster_centers_), columns=mkt)
        names = {}
        used = set()
        q = centers["vol_21"].idxmin()
        names[q] = "Defensive"
        used.add(q)
        remaining = [i for i in centers.index if i not in used]
        if remaining:
            g2 = centers.loc[remaining, "mom_6m"].idxmax()
            names[g2] = "Growth"
            used.add(g2)
        for i in centers.index:
            if i not in used:
                names[i] = "High-Risk"
        clust["profile"] = clust["cluster"].map(names)
    else:
        clust["cluster"] = 0
        clust["profile"] = "N/A"

    return metrics_df, fi_df, latest, clust


# ============================================================
# TOOL 4: PREDICTIVE MODEL (Daily direction)
# ============================================================

def run_predictive_model(prices):
    """Random Forest classifier for next-day direction prediction."""
    results, all_fi = [], []
    for ticker in prices["ticker"].unique():
        sub = prices[prices["ticker"] == ticker][["data", "preco"]].copy()
        sub = sub.sort_values("data").set_index("data")
        if len(sub) < 100:
            continue
        df = pd.DataFrame()
        df["preco"] = sub["preco"]
        df["retorno"] = df["preco"].pct_change()
        df["ret_1d"] = df["retorno"].shift(1)
        df["ret_5d"] = df["preco"].pct_change(5)
        df["ret_20d"] = df["preco"].pct_change(20)
        df["vol_20d"] = df["retorno"].rolling(20).std()
        df["above_ma20"] = np.where(
            df["preco"] > df["preco"].rolling(20).mean(), 1, 0)
        df["target"] = np.where(df["retorno"].shift(-1) > 0, 1, 0)
        df = df.dropna()
        if len(df) < 50:
            continue
        feats = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "above_ma20"]
        X, y = df[feats], df["target"]
        split = int(len(df) * 0.7)
        m = RandomForestClassifier(
            n_estimators=300, max_depth=4, random_state=42, n_jobs=-1)
        m.fit(X.iloc[:split], y.iloc[:split])
        y_pred = m.predict(X.iloc[split:])
        y_prob = m.predict_proba(X.iloc[split:])[:, 1]
        acc = accuracy_score(y.iloc[split:], y_pred)
        auc = (roc_auc_score(y.iloc[split:], y_prob)
               if len(set(y.iloc[split:])) > 1 else np.nan)
        results.append({
            "ticker": ticker,
            "accuracy": round(acc, 4),
            "auc": round(auc, 4) if pd.notna(auc) else np.nan,
            "n_test": len(y) - split,
        })
        all_fi.append(pd.Series(m.feature_importances_, index=feats, name=ticker))
    return pd.DataFrame(results), (pd.DataFrame(all_fi) if all_fi else pd.DataFrame())


# ============================================================
# TOOL 5: TRADING SYSTEM (MA Crossover)
# ============================================================

def run_trading_system(prices, ma_short=20, ma_long=60):
    """Moving average crossover strategy with backtest."""
    all_res, summary = [], []
    for ticker in prices["ticker"].unique():
        sub = prices[prices["ticker"] == ticker][["data", "preco"]].copy()
        sub = sub.sort_values("data")
        if len(sub) < ma_long + 10:
            continue
        df = pd.DataFrame({"data": sub["data"].values, "preco": sub["preco"].values})
        df["ma_short"] = df["preco"].rolling(ma_short).mean()
        df["ma_long"] = df["preco"].rolling(ma_long).mean()
        df["signal"] = np.where(df["ma_short"] > df["ma_long"], 1, 0)
        df["ret_acao"] = df["preco"].pct_change()
        df["ret_est"] = df["signal"].shift(1) * df["ret_acao"]
        df = df.dropna()
        df["acum_acao"] = (1 + df["ret_acao"]).cumprod()
        df["acum_est"] = (1 + df["ret_est"]).cumprod()
        df["ticker"] = ticker
        all_res.append(df)
        rb = df["acum_acao"].iloc[-1] - 1
        rs = df["acum_est"].iloc[-1] - 1
        summary.append({
            "ticker": ticker,
            "ret_buyhold": round(rb * 100, 1),
            "ret_estrategia": round(rs * 100, 1),
            "alpha": round((rs - rb) * 100, 1),
        })
    curves = pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame()
    return curves, pd.DataFrame(summary)


# ============================================================
# TOOL 6: MONTE CARLO (Portfolio Optimization)
# ============================================================

def run_monte_carlo(prices, n_sim=10000):
    """Monte Carlo simulation for efficient frontier."""
    wide = prices.pivot_table(index="data", columns="ticker", values="preco")
    wide = wide.dropna(axis=1, how="all").dropna()
    tickers_ok = wide.columns.tolist()
    if len(tickers_ok) < 2:
        return pd.DataFrame(), {}, tickers_ok
    rets = wide.pct_change().dropna()
    mean_ret, cov_mat = rets.mean(), rets.cov()
    results = []
    for _ in range(n_sim):
        w = np.random.random(len(tickers_ok))
        w = w / w.sum()
        ret_a = np.dot(w, mean_ret) * 252
        risk_a = np.sqrt(np.dot(w.T, np.dot(cov_mat * 252, w)))
        sharpe = ret_a / risk_a if risk_a > 0 else 0
        results.append(list(w) + [ret_a, risk_a, sharpe])
    cols = [f"w_{t}" for t in tickers_ok] + ["retorno", "risco", "sharpe"]
    df = pd.DataFrame(results, columns=cols)
    best = df.loc[df["sharpe"].idxmax()]
    best_dict = {
        "sharpe": best["sharpe"],
        "retorno": best["retorno"],
        "risco": best["risco"],
    }
    for t in tickers_ok:
        best_dict[t] = best[f"w_{t}"]
    return df, best_dict, tickers_ok


# ============================================================
# TOOL 7: COMPOSITE SCORING
# ============================================================

PROFILES = {
    "Conservative": {"market": 0.60, "quality": 0.40},
    "Moderate": {"market": 0.50, "quality": 0.50},
    "Aggressive": {"market": 0.40, "quality": 0.60},
}


def run_scoring(prices, fund_df, profile="Moderate"):
    """Composite score combining market + quality factors."""
    weights = PROFILES.get(profile, PROFILES["Moderate"])

    snap = prices.dropna(subset=["vol_21", "mom_6m"]).sort_values("data")
    snap = snap.groupby("ticker").tail(1).copy()

    fund_cols = [c for c in [
        "ticker", "P_L", "P_VP", "EV_EBITDA", "nome", "setor",
        "marketCap", "div_yield", "profitMargins", "returnOnEquity",
        "revenueGrowth", "ebitdaMargins",
    ] if c in fund_df.columns]
    snap = snap.merge(fund_df[fund_cols], on="ticker", how="left")

    # Market score
    snap["sc_mom"] = snap["mom_6m"].rank(pct=True) * 100
    snap["sc_vol"] = snap["vol_21"].rank(pct=True, ascending=False) * 100
    snap["sc_dd"] = snap["drawdown"].rank(pct=True, ascending=False) * 100
    snap["score_market"] = (
        snap["sc_mom"] * 0.5
        + snap["sc_vol"] * 0.3
        + snap["sc_dd"] * 0.2
    )

    # Quality score (lower P/VP is better)
    if "P_VP" in snap.columns and snap["P_VP"].notna().sum() > 0:
        snap["score_quality"] = snap["P_VP"].rank(
            pct=True, ascending=True) * 100
    else:
        snap["score_quality"] = 50.0
    snap["score_quality"] = snap["score_quality"].fillna(50)

    # Composite
    snap["score"] = (
        snap["score_market"] * weights["market"]
        + snap["score_quality"] * weights["quality"]
    ).round(1)

    def rec(s):
        if pd.isna(s):
            return "HOLD"
        if s >= 65:
            return "BUY"
        if s >= 35:
            return "HOLD"
        return "SELL"

    snap["recommendation"] = snap["score"].apply(rec)
    snap["rank"] = snap["score"].rank(ascending=False, method="min").astype(int)

    # Ensure display columns exist
    for col in ["nome", "setor", "P_L", "P_VP", "EV_EBITDA", "marketCap"]:
        if col not in snap.columns:
            snap[col] = "-" if col in ["nome", "setor"] else np.nan

    # Clean string columns
    for col in ["nome", "setor"]:
        if col in snap.columns:
            snap[col] = snap[col].fillna("-").replace("", "-")
            snap[col] = snap[col].astype(str).str.strip()
            snap[col] = snap[col].apply(lambda x: "-" if x in ("", "?", "None", "nan") else x)

    return snap.sort_values("rank")
