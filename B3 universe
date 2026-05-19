"""
b3_universe.py - Curated B3 stock universe database
~80 most liquid Brazilian stocks organized by sector
"""

# Curated universe of B3 stocks: {ticker: (name, sector)}
B3_UNIVERSE = {
    # ===== FINANCIAL SERVICES =====
    "ITUB4.SA":  ("Itau Unibanco", "Financial Services"),
    "BBDC4.SA":  ("Bradesco", "Financial Services"),
    "BBAS3.SA":  ("Banco do Brasil", "Financial Services"),
    "SANB11.SA": ("Santander BR", "Financial Services"),
    "BPAC11.SA": ("BTG Pactual", "Financial Services"),
    "ABCB4.SA":  ("Banco ABC Brasil", "Financial Services"),
    "ITSA4.SA":  ("Itausa", "Financial Services"),
    "B3SA3.SA":  ("B3 (Bolsa Brasil)", "Financial Services"),
    "BBSE3.SA":  ("BB Seguridade", "Financial Services"),
    "PSSA3.SA":  ("Porto Seguro", "Financial Services"),
    "IRBR3.SA":  ("IRB Brasil RE", "Financial Services"),

    # ===== ENERGY (Oil & Gas) =====
    "PETR4.SA":  ("Petrobras PN", "Energy"),
    "PETR3.SA":  ("Petrobras ON", "Energy"),
    "PRIO3.SA":  ("PRIO (PetroRio)", "Energy"),
    "RECV3.SA":  ("PetroReconcavo", "Energy"),
    "RRRP3.SA":  ("3R Petroleum", "Energy"),
    "VBBR3.SA":  ("Vibra Energia", "Energy"),
    "UGPA3.SA":  ("Ultrapar", "Energy"),

    # ===== UTILITIES =====
    "EGIE3.SA":  ("Engie Brasil", "Utilities"),
    "EQTL3.SA":  ("Equatorial", "Utilities"),
    "CMIG4.SA":  ("Cemig", "Utilities"),
    "ELET3.SA":  ("Eletrobras ON", "Utilities"),
    "ELET6.SA":  ("Eletrobras PNB", "Utilities"),
    "SBSP3.SA":  ("Sabesp", "Utilities"),
    "TAEE11.SA": ("Taesa", "Utilities"),
    "CPFE3.SA":  ("CPFL Energia", "Utilities"),
    "ENGI11.SA": ("Energisa", "Utilities"),
    "NEOE3.SA":  ("Neoenergia", "Utilities"),
    "AURE3.SA":  ("Auren Energia", "Utilities"),

    # ===== BASIC MATERIALS (Mining, Steel, Paper) =====
    "VALE3.SA":  ("Vale", "Basic Materials"),
    "SUZB3.SA":  ("Suzano", "Basic Materials"),
    "KLBN11.SA": ("Klabin", "Basic Materials"),
    "CSNA3.SA":  ("CSN", "Basic Materials"),
    "USIM5.SA":  ("Usiminas", "Basic Materials"),
    "GGBR4.SA":  ("Gerdau", "Basic Materials"),
    "GOAU4.SA":  ("Metalurgica Gerdau", "Basic Materials"),
    "BRAP4.SA":  ("Bradespar", "Basic Materials"),

    # ===== CONSUMER DEFENSIVE (Food, Beverages) =====
    "ABEV3.SA":  ("Ambev", "Consumer Defensive"),
    "JBSS3.SA":  ("JBS", "Consumer Defensive"),
    "MRFG3.SA":  ("Marfrig", "Consumer Defensive"),
    "BRFS3.SA":  ("BRF (Sadia/Perdigao)", "Consumer Defensive"),
    "BEEF3.SA":  ("Minerva Foods", "Consumer Defensive"),
    "SMTO3.SA":  ("Sao Martinho", "Consumer Defensive"),
    "NTCO3.SA":  ("Natura &Co", "Consumer Defensive"),

    # ===== INDUSTRIALS =====
    "WEGE3.SA":  ("WEG", "Industrials"),
    "EMBR3.SA":  ("Embraer", "Industrials"),
    "RAIL3.SA":  ("Rumo (Cosan Logistica)", "Industrials"),
    "CCRO3.SA":  ("CCR", "Industrials"),
    "AZUL4.SA":  ("Azul", "Industrials"),
    "GOLL4.SA":  ("Gol Linhas Aereas", "Industrials"),
    "TUPY3.SA":  ("Tupy", "Industrials"),
    "POMO4.SA":  ("Marcopolo", "Industrials"),
    "ECOR3.SA":  ("EcoRodovias", "Industrials"),

    # ===== CONSUMER CYCLICAL (Retail, Auto) =====
    "MGLU3.SA":  ("Magazine Luiza", "Consumer Cyclical"),
    "LREN3.SA":  ("Lojas Renner", "Consumer Cyclical"),
    "RENT3.SA":  ("Localiza", "Consumer Cyclical"),
    "VAMO3.SA":  ("Vamos Locacao", "Consumer Cyclical"),
    "AMER3.SA":  ("Americanas", "Consumer Cyclical"),
    "PETZ3.SA":  ("Petz", "Consumer Cyclical"),
    "ALPA4.SA":  ("Alpargatas", "Consumer Cyclical"),
    "ARZZ3.SA":  ("Arezzo", "Consumer Cyclical"),
    "GUAR3.SA":  ("Guararapes (Riachuelo)", "Consumer Cyclical"),

    # ===== HEALTHCARE =====
    "RDOR3.SA":  ("Rede DOr", "Healthcare"),
    "HAPV3.SA":  ("Hapvida", "Healthcare"),
    "QUAL3.SA":  ("Qualicorp", "Healthcare"),
    "FLRY3.SA":  ("Fleury", "Healthcare"),
    "HYPE3.SA":  ("Hypera Pharma", "Healthcare"),

    # ===== COMMUNICATION =====
    "VIVT3.SA":  ("Vivo (Telefonica)", "Communication Services"),
    "TIMS3.SA":  ("TIM Brasil", "Communication Services"),

    # ===== TECHNOLOGY =====
    "TOTS3.SA":  ("Totvs", "Technology"),
    "POSI3.SA":  ("Positivo", "Technology"),
    "LWSA3.SA":  ("Locaweb", "Technology"),

    # ===== REAL ESTATE =====
    "MRVE3.SA":  ("MRV Engenharia", "Real Estate"),
    "CYRE3.SA":  ("Cyrela", "Real Estate"),
    "EZTC3.SA":  ("EzTec", "Real Estate"),
    "DIRR3.SA":  ("Direcional", "Real Estate"),
    "MULT3.SA":  ("Multiplan", "Real Estate"),
    "IGTI11.SA": ("Iguatemi", "Real Estate"),
    "BRML3.SA":  ("BR Malls", "Real Estate"),
}

# Predefined portfolios for quick selection
PORTFOLIOS = {
    "🇧🇷 Ibovespa Top 10": [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
        "BBAS3.SA", "WEGE3.SA", "B3SA3.SA", "ITSA4.SA", "RENT3.SA",
    ],
    "🏦 Bancos & Financeiras": [
        "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA",
        "BPAC11.SA", "ABCB4.SA", "ITSA4.SA", "B3SA3.SA", "BBSE3.SA",
    ],
    "⚡ Energia & Utilities": [
        "PETR4.SA", "PRIO3.SA", "VBBR3.SA",
        "EGIE3.SA", "EQTL3.SA", "CMIG4.SA", "ELET3.SA", "SBSP3.SA", "TAEE11.SA",
    ],
    "⛏️ Materiais Básicos": [
        "VALE3.SA", "SUZB3.SA", "KLBN11.SA",
        "CSNA3.SA", "USIM5.SA", "GGBR4.SA",
    ],
    "🛒 Consumo Defensivo": [
        "ABEV3.SA", "JBSS3.SA", "MRFG3.SA",
        "BRFS3.SA", "BEEF3.SA", "NTCO3.SA",
    ],
    "🚗 Consumo Cíclico": [
        "RENT3.SA", "LREN3.SA", "MGLU3.SA",
        "VAMO3.SA", "PETZ3.SA", "ARZZ3.SA",
    ],
    "🏭 Industriais": [
        "WEGE3.SA", "EMBR3.SA", "RAIL3.SA",
        "CCRO3.SA", "ECOR3.SA", "TUPY3.SA",
    ],
    "💊 Saúde": [
        "RDOR3.SA", "HAPV3.SA", "QUAL3.SA", "FLRY3.SA", "HYPE3.SA",
    ],
    "📱 Tech & Telecom": [
        "VIVT3.SA", "TIMS3.SA", "TOTS3.SA", "LWSA3.SA",
    ],
    "🏠 Real Estate": [
        "MRVE3.SA", "CYRE3.SA", "EZTC3.SA", "MULT3.SA", "IGTI11.SA",
    ],
}


def get_all_sectors():
    """Return sorted list of unique sectors."""
    return sorted(set(s for _, s in B3_UNIVERSE.values()))


def get_tickers_by_sector(sector):
    """Return list of tickers in a given sector."""
    return [t for t, (_, s) in B3_UNIVERSE.items() if s == sector]


def get_ticker_label(ticker):
    """Return display label: 'TICKER - Name (Sector)'"""
    if ticker not in B3_UNIVERSE:
        return ticker.replace(".SA", "")
    name, sector = B3_UNIVERSE[ticker]
    return f"{ticker.replace('.SA','')} - {name}"


def get_universe_df():
    """Return universe as DataFrame for display."""
    import pandas as pd
    data = [
        {"Ticker": t.replace(".SA", ""), "Name": n, "Sector": s}
        for t, (n, s) in B3_UNIVERSE.items()
    ]
    return pd.DataFrame(data).sort_values(["Sector", "Ticker"]).reset_index(drop=True)
