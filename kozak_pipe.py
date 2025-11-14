import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


epsilon = 1e-8


# ----------------------------------------------------------------------
# 0. Panel construction
# ----------------------------------------------------------------------

def build_panel_matrices(df, char_cols,
                         ret_col="ret",
                         id_col="permno",
                         date_col="date"):
    """
    From the long panel df → lists of (X_t, z_t, ids_t) over t.
    X_t: (N_t, J) with first column = constant
    z_t: (N_t,)
    """
    df = df.sort_values([date_col, id_col])

    dates = df[date_col].drop_duplicates().sort_values().tolist()

    X_list = []
    Z_list = []
    ids_list = []

    for d in dates:
        tmp = df[df[date_col] == d].copy()

        # fill missing within month using cross-sectional medians
        for col in [ret_col] + char_cols:
            try:
                median_val = tmp[col].median()
            except Exception:
                median_val = 0.0
            if pd.isna(median_val):
                median_val = 0.0
            tmp[col] = tmp[col].fillna(median_val)

        if tmp.empty:
            # if everything is NaN for that month, skip
            continue

        ids = tmp[id_col].values           # (N_t,)
        z = tmp[ret_col].values            # (N_t,)
        X = tmp[char_cols].values          # (N_t, J_char)

        # Add constant column (intercept)
        const = np.ones((X.shape[0], 1))   # (N_t, 1)
        X_full = np.hstack([const, X])     # (N_t, 1+J)

        ids_list.append(ids)
        Z_list.append(z)
        X_list.append(X_full)

    return dates, ids_list, Z_list, X_list


# ----------------------------------------------------------------------
# 1. Factor construction (OLS / univariate)
# ----------------------------------------------------------------------

def compute_ols_factors(Z_list, X_list, ridge=1e-6):
    """
    Cross-sectional OLS factors per period:
        f_t = (X_t' X_t)^{-1} X_t' z_t
    Returns F_ols: (T, J)
    """
    T = len(Z_list)
    J = X_list[0].shape[1]
    F_ols = np.zeros((T, J))

    for t in range(T):
        z_t = Z_list[t].reshape(-1, 1)   # (N_t, 1)
        X_t = X_list[t]                  # (N_t, J)

        XtX = X_t.T @ X_t
        XtX_reg = XtX + ridge * np.eye(J)
        XtZ = X_t.T @ z_t

        beta_t = np.linalg.solve(XtX_reg, XtZ)  # (J, 1)
        F_ols[t, :] = beta_t.ravel()

    return F_ols


def compute_univariate_factors(Z_list, X_list, ridge=1e-6):
    """
    For each characteristic j≥1, regress z_t on [1, x_{j,t}] and
    take the slope as the factor return.

    Returns F_uni: (T, J) with col 0 = constant-only factor,
    cols 1..J-1 = univariate characteristic factors.
    """
    T = len(Z_list)
    J = X_list[0].shape[1]
    F_uni = np.zeros((T, J))

    for t in range(T):
        z_t = Z_list[t].reshape(-1, 1)   # (N_t, 1)
        X_t = X_list[t]                  # (N_t, J)

        # constant-only regression
        ones = X_t[:, [0]]
        XtX0 = ones.T @ ones + ridge * np.eye(1)
        XtZ0 = ones.T @ z_t
        beta0 = np.linalg.solve(XtX0, XtZ0)
        F_uni[t, 0] = float(beta0)

        # univariate per characteristic
        for j in range(1, J):
            X_j = X_t[:, [0, j]]  # constant + char j
            XtX = X_j.T @ X_j + ridge * np.eye(2)
            XtZ = X_j.T @ z_t
            beta_j = np.linalg.solve(XtX, XtZ)
            F_uni[t, j] = float(beta_j[1, 0])

    return F_uni


# ----------------------------------------------------------------------
# 2. Hedging characteristics (cross-sectional residualization)
# ----------------------------------------------------------------------

def hedge_characteristics_X(X_list, ridge=1e-6, include_const=True):
    """
    For each t and each j>=1, regress x_{j,t} on other columns of X_t
    (optionally including the constant) and take residuals.

    X_list: list of (N_t, J) arrays. Column 0 is constant.
    Returns H_list with same shapes.
    """
    H_list = []

    for X_t in X_list:
        X_t = np.asarray(X_t)
        N_t, J = X_t.shape
        H_t = np.zeros_like(X_t)

        # keep constant
        H_t[:, 0] = X_t[:, 0]

        for j in range(1, J):
            x_j = X_t[:, [j]]

            mask = np.ones(J, dtype=bool)
            mask[j] = False
            if not include_const:
                mask[0] = False
            X_minus = X_t[:, mask]

            # protect against NaNs, infs
            good = np.all(np.isfinite(X_minus), axis=1) & np.isfinite(x_j.ravel())
            X_use = X_minus[good]
            y_use = x_j[good]

            XtX = X_use.T @ X_use
            XtX_reg = XtX + ridge * np.eye(XtX.shape[0])
            Xt_y = X_use.T @ y_use

            coeff = np.linalg.solve(XtX_reg, Xt_y)
            proj = X_minus @ coeff

            H_t[:, j] = (x_j - proj).ravel()

        H_list.append(H_t)

    return H_list


# ----------------------------------------------------------------------
# 3. GLS factors with time-varying diagonal Sigma_t
# ----------------------------------------------------------------------

def build_rolling_sigma_diag_list(df, dates, ids_list,
                                  ret_col="ret",
                                  id_col="permno",
                                  date_col="date",
                                  window=36,
                                  min_periods=12):
    """
    Build list of sigma_{i,t}^2 (variances) aligned with ids_list.

    Use rolling window volatility of returns per stock:
        sigma_{i,t} = rolling std over 'window' months at t.
    """
    df = df.sort_values([id_col, date_col]).copy()

    df["sigma_roll"] = (
        df.groupby(id_col)[ret_col]
          .rolling(window=window, min_periods=min_periods)
          .std(ddof=1)
          .reset_index(level=0, drop=True)
    )

    overall_sigma = df[ret_col].std(ddof=1)
    df["sigma_roll"] = df["sigma_roll"].fillna(overall_sigma)

    sigma_panel = df.set_index([date_col, id_col])["sigma_roll"]

    Sigma_diag_list = []

    for d, ids_t in zip(dates, ids_list):
        mi = pd.MultiIndex.from_arrays(
            [np.repeat(d, len(ids_t)), ids_t],
            names=[date_col, id_col],
        )
        sig_t = sigma_panel.reindex(mi).to_numpy()

        bad = ~np.isfinite(sig_t)
        if bad.any():
            med_t = np.nanmedian(sig_t)
            if not np.isfinite(med_t):
                med_t = overall_sigma
            sig_t[bad] = med_t

        Sigma_diag_list.append(sig_t ** 2)

    return Sigma_diag_list


def compute_gls_factors_diagSigma_t(Z_list, X_list, Sigma_diag_list,
                                    ridge_beta=1e-6,
                                    min_var=1e-8):
    """
    GLS characteristic factors with time-varying diagonal Sigma_t:

        f_t = (X_t' W_t X_t)^{-1} X_t' W_t z_t,
        W_t = diag(1 / sigma_{i,t}^2).

    Returns F_gls: (T, J)
    """
    T = len(Z_list)
    J = X_list[0].shape[1]
    F_gls = np.zeros((T, J))

    for t in range(T):
        z_t = np.asarray(Z_list[t]).reshape(-1, 1)  # (N_t, 1)
        X_t = np.asarray(X_list[t])                 # (N_t, J)
        sigma2_t = np.asarray(Sigma_diag_list[t])   # (N_t,)

        sigma2_t[~np.isfinite(sigma2_t)] = np.nan
        if np.isnan(sigma2_t).all():
            raise ValueError(f"All variances NaN at t={t}")
        nan_mask = np.isnan(sigma2_t)
        if nan_mask.any():
            med = np.nanmedian(sigma2_t)
            sigma2_t[nan_mask] = med

        sigma2_t = np.maximum(sigma2_t, min_var)

        w_t = 1.0 / sigma2_t
        W_half = np.sqrt(w_t).reshape(-1, 1)

        Xw = X_t * W_half
        zw = z_t * W_half

        XtWX = Xw.T @ Xw
        XtWX_reg = XtWX + ridge_beta * np.eye(J)
        XtWz = Xw.T @ zw

        beta_t = np.linalg.solve(XtWX_reg, XtWz)
        F_gls[t, :] = beta_t.ravel()

    return F_gls


# ----------------------------------------------------------------------
# 4. Sharpe in the factor span
# ----------------------------------------------------------------------

def span_sharpe_from_factors(F, freq=1, ridge=1e-6):
    """
    Maximum Sharpe ratio and squared Sharpe ratio in the span of factor
    returns F:

        SR^2 = mu_f' Sigma_f^{-1} mu_f

    freq : periods per year (1 → per-period, 12 → annualized monthly).
    """
    F = np.asarray(F)
    if F.ndim == 1:
        F = F.reshape(-1, 1)

    # drop rows with any non-finite
    mask = np.all(np.isfinite(F), axis=1)
    F = F[mask]
    T, K = F.shape

    if T == 0 or K == 0:
        raise ValueError("Empty factor matrix after cleaning.")

    mu_f = F.mean(axis=0).reshape(-1, 1)

    if K == 1:
        var_f = float(np.var(F, ddof=1)) + ridge
        sr2_period = float(mu_f[0, 0] ** 2 / var_f)
    else:
        Sigma_f = np.cov(F, rowvar=False)
        Sigma_f = 0.5 * (Sigma_f + Sigma_f.T) + ridge * np.eye(K)

        eigvals, eigvecs = np.linalg.eigh(Sigma_f)
        eigvals_clipped = np.maximum(eigvals, ridge)
        Sigma_inv = (eigvecs / eigvals_clipped) @ eigvecs.T

        sr2_period = float(mu_f.T @ Sigma_inv @ mu_f)

    sr2 = freq * sr2_period
    sr = np.sqrt(sr2)
    return sr, sr2


# ----------------------------------------------------------------------
# 5. PCA truncation
# ----------------------------------------------------------------------
def pca_truncate(F, K):
    """
    Take factor returns F (T × J), return top K principal component factors (T × K).
    """
    F = np.asarray(F)
    T, J = F.shape

    # Covariance of factors (np.cov centers internally)
    Sigma_f = np.cov(F.T)              # J×J

    # Eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(Sigma_f)
    idx = np.argsort(eigvals)[::-1]    # sort by eigenvalue desc
    eigvecs = eigvecs[:, idx]          # J×J

    eigvecs_K = eigvecs[:, :K]         # J×K

    # IMPORTANT: project the original F, not the centered one
    F_pca = F @ eigvecs_K              # T×K

    return F_pca


# ----------------------------------------------------------------------
# 6. Top-level pipeline: takes ONLY df, does everything else internally
# ----------------------------------------------------------------------

def run_kozak(df):
    """
    High-level helper:
      - assumes df has columns: permno, date, price, age, re/ret, plus chars
      - returns the SR^2 vs K DataFrames matching your notebook plots.

    Returns a dict:
      {
        "df_ols_sr2":  DataFrame (K × {Unhedged, Hedged 1x, Hedged 2x, Hedged 3x, GLS (pca)}),
        "df_ols_impr": DataFrame (K × improvements vs Unhedged),
        "df_uni_sr2":  same for univariate factors,
        "df_uni_impr": same improvements for univariate,
        "meta":        {dates, ids_list, X_list, F_ols_*, F_gls, ...}
      }
    """

    df = df.copy()

    # Rename if needed (same as notebook)
    if "re" in df.columns and "ret" not in df.columns:
        df = df.rename(columns={"re": "ret"})

    # Parse date as in notebook: "%m/%Y"
    df["date"] = pd.to_datetime(df["date"], format="%m/%Y")
    df = df.sort_values(["date", "permno"]).reset_index(drop=True)

    # Character columns (everything except meta + ret)
    meta_cols = ["permno", "date"]
    ret_col = "ret"
    char_cols = [c for c in df.columns if c not in meta_cols + [ret_col]]

    # 1. panel matrices
    dates, ids_list, Z_list, X_list = build_panel_matrices(
        df, char_cols, ret_col=ret_col, id_col="permno", date_col="date"
    )

    # 2. OLS factors for 0x,1x,2x,3x hedged
    F_ols_0 = compute_ols_factors(Z_list, X_list)

    X1_list = hedge_characteristics_X(X_list)
    F_ols_1 = compute_ols_factors(Z_list, X1_list)

    X2_list = hedge_characteristics_X(X1_list)
    F_ols_2 = compute_ols_factors(Z_list, X2_list)

    X3_list = hedge_characteristics_X(X2_list)
    F_ols_3 = compute_ols_factors(Z_list, X3_list)

    # 3. Univariate factors for 0x..3x hedged
    F_uni_0 = compute_univariate_factors(Z_list, X_list)
    F_uni_1 = compute_univariate_factors(Z_list, X1_list)
    F_uni_2 = compute_univariate_factors(Z_list, X2_list)
    F_uni_3 = compute_univariate_factors(Z_list, X3_list)

    # 4. GLS factors (time-varying diagonal Sigma_t)
    Sigma_diag_list = build_rolling_sigma_diag_list(
        df, dates, ids_list,
        ret_col=ret_col, id_col="permno", date_col="date"
    )
    F_gls = compute_gls_factors_diagSigma_t(Z_list, X_list, Sigma_diag_list)

    # 5. SR^2 vs K (annualized: freq=12 for monthly data)
    K_max = 55
    freq = 12
    Ks = np.arange(2, K_max + 1)

    ols_unhedged_sr2 = []
    ols_h1_sr2 = []
    ols_h2_sr2 = []
    ols_h3_sr2 = []
    ols_gls_sr2 = []

    uni_unhedged_sr2 = []
    uni_h1_sr2 = []
    uni_h2_sr2 = []
    uni_h3_sr2 = []
    uni_gls_sr2 = []

    for K in Ks:
        # OLS
        F0K = pca_truncate(F_ols_0, K)
        F1K = pca_truncate(F_ols_1, K)
        F2K = pca_truncate(F_ols_2, K)
        F3K = pca_truncate(F_ols_3, K)
        FGK = pca_truncate(F_gls,    K)

        _, sr2_ols0 = span_sharpe_from_factors(F0K, freq=freq)
        _, sr2_ols1 = span_sharpe_from_factors(F1K, freq=freq)
        _, sr2_ols2 = span_sharpe_from_factors(F2K, freq=freq)
        _, sr2_ols3 = span_sharpe_from_factors(F3K, freq=freq)
        _, sr2_olsg = span_sharpe_from_factors(FGK, freq=freq)

        ols_unhedged_sr2.append(sr2_ols0)
        ols_h1_sr2.append(sr2_ols1)
        ols_h2_sr2.append(sr2_ols2)
        ols_h3_sr2.append(sr2_ols3)
        ols_gls_sr2.append(sr2_olsg)

        # Univariate (same GLS benchmark)
        U0K = pca_truncate(F_uni_0, K)
        U1K = pca_truncate(F_uni_1, K)
        U2K = pca_truncate(F_uni_2, K)
        U3K = pca_truncate(F_uni_3, K)
        UGK = pca_truncate(F_gls,    K)

        _, sr2_uni0 = span_sharpe_from_factors(U0K, freq=freq)
        _, sr2_uni1 = span_sharpe_from_factors(U1K, freq=freq)
        _, sr2_uni2 = span_sharpe_from_factors(U2K, freq=freq)
        _, sr2_uni3 = span_sharpe_from_factors(U3K, freq=freq)
        _, sr2_unig = span_sharpe_from_factors(UGK, freq=freq)

        uni_unhedged_sr2.append(sr2_uni0)
        uni_h1_sr2.append(sr2_uni1)
        uni_h2_sr2.append(sr2_uni2)
        uni_h3_sr2.append(sr2_uni3)
        uni_gls_sr2.append(sr2_unig)

    # Level plots (like “Average total SR^2”)
    df_ols_sr2 = pd.DataFrame({
        "Unhedged":  ols_unhedged_sr2,
        "Hedged 1x": ols_h1_sr2,
        "Hedged 2x": ols_h2_sr2,
        "Hedged 3x": ols_h3_sr2,
        "GLS (pca)": ols_gls_sr2,
    }, index=Ks)

    df_uni_sr2 = pd.DataFrame({
        "Unhedged":  uni_unhedged_sr2,
        "Hedged 1x": uni_h1_sr2,
        "Hedged 2x": uni_h2_sr2,
        "Hedged 3x": uni_h3_sr2,
        "GLS (pca)": uni_gls_sr2,
    }, index=Ks)

    # % improvement vs unhedged
    ou = np.array(ols_unhedged_sr2)
    uu = np.array(uni_unhedged_sr2)

    df_ols_impr = pd.DataFrame({
        "Hedged 1x": 100 * (np.array(ols_h1_sr2) - ou) / ou,
        "Hedged 2x": 100 * (np.array(ols_h2_sr2) - ou) / ou,
        "Hedged 3x": 100 * (np.array(ols_h3_sr2) - ou) / ou,
        "GLS (pca)": 100 * (np.array(ols_gls_sr2) - ou) / ou,
    }, index=Ks)

    df_uni_impr = pd.DataFrame({
        "Hedged 1x": 100 * (np.array(uni_h1_sr2) - uu) / uu,
        "Hedged 2x": 100 * (np.array(uni_h2_sr2) - uu) / uu,
        "Hedged 3x": 100 * (np.array(uni_h3_sr2) - uu) / uu,
        "GLS (pca)": 100 * (np.array(uni_gls_sr2) - uu) / uu,
    }, index=Ks)

    return {
        "df_ols_sr2": df_ols_sr2,
        "df_ols_impr": df_ols_impr,
        "df_uni_sr2": df_uni_sr2,
        "df_uni_impr": df_uni_impr,
        "meta": {
            "dates": dates,
            "ids_list": ids_list,
            "Z_list": Z_list,
            "X_list": X_list,
            "F_ols_0": F_ols_0,
            "F_ols_1": F_ols_1,
            "F_ols_2": F_ols_2,
            "F_ols_3": F_ols_3,
            "F_uni_0": F_uni_0,
            "F_uni_1": F_uni_1,
            "F_uni_2": F_uni_2,
            "F_uni_3": F_uni_3,
            "F_gls": F_gls,
        },
    }
