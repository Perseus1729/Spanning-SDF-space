def sic_to_industry(sic):
    # handle missing / non-numeric
    try:
        sic = int(float(sic))
    except (ValueError, TypeError):
        return np.nan

    if 1 <= sic <= 999: return "Agriculture"
    if 1000 <= sic <= 1499: return "Mining"
    if 1500 <= sic <= 1799: return "Construction"
    if 2000 <= sic <= 3999: return "Manufacturing"
    if 4000 <= sic <= 4999: return "Transport_Utilities"
    if 5000 <= sic <= 5199: return "Wholesale"
    if 5200 <= sic <= 5999: return "Retail"
    if 6000 <= sic <= 6799: return "Finance_Insurance_RE"
    if 7000 <= sic <= 8999: return "Services"
    if 9000 <= sic <= 9999: return "Public_Admin"
    return "Other"

def load_data(data_path: str):
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
    cols_needed = ["permno","permco","date","siccd","industry","prc","vol","ret","retx","shrout","mkt","ticker","comnam","year","month",'vwretd']
    present = [c for c in cols_needed if c in df.columns]
    df = df[present].copy()
    if {'prc','shrout'}.issubset(df.columns):
        df['mktcap'] = df['prc'].abs() * df['shrout'] * 1000.0
    df['industry'] = df['siccd'].apply(sic_to_industry) if 'siccd' in df.columns else np.nan
    df['year'] = df['date'].dt.year
    if 'ret' in df.columns:
        df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    if 'vwretd' in df.columns:
        df['vwretd'] = pd.to_numeric(df['vwretd'], errors='coerce')
    return df

DATA_PATH = "MSF_1996_2023.csv"
RF_PATH = "F-F_Research_Data_Factors.csv"  # optional RF file (percent per month), columns: date, rf


msf = load_data(DATA_PATH)
msf.head()

# Merge into your CRSP MSF
msf['year'] = msf['date'].dt.year
msf['month'] = msf['date'].dt.month

tot_df = msf