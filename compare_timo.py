import math
import numpy as np
import pandas as pd

from solver import solve_timoshenko_membrane


def read_excel_profile(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read Excel-exported CSV and return (r, h) arrays sorted by h ascending.

    We use the last pair of columns named exactly '-x' and 'h' (the sheet has multiple parts).
    The file uses semicolon separators and comma decimals.
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",", engine="python")

    # Identify the last occurrences of '-x' and 'h'
    def _norm(name: object) -> str:
        s = str(name).replace("\ufeff", "").strip()
        # Pandas makes duplicate headers like 'h', 'h.1', ...
        if "." in s:
            s = s.split(".", 1)[0]
        return s

    idx_x = [i for i, c in enumerate(df.columns) if _norm(c) == "-x"]
    idx_h = [i for i, c in enumerate(df.columns) if _norm(c) == "h"]
    if not idx_x or not idx_h:
        # Fallback: search header row manually in a header-less read
        df_raw = pd.read_csv(csv_path, sep=';', decimal=',', header=None, engine='python')
        found = False
        rows_to_scan = min(30, len(df_raw))
        cols_to_scan = df_raw.shape[1]
        def _norm_cell(val: object) -> str:
            try:
                s = str(val).replace("\ufeff", "").strip()
            except Exception:
                s = ""
            if "." in s:
                s = s.split(".", 1)[0]
            return s
        # Collect all candidate (-x,h) pairs; choose the one with the most numeric rows (full curve)
        candidates: list[tuple[int, int, int]] = []  # (row, col_x, count)
        for r in range(rows_to_scan):
            for c in range(max(0, cols_to_scan - 2)):
                name_c = _norm_cell(df_raw.iat[r, c])
                name_n = _norm_cell(df_raw.iat[r, c + 1])
                if (name_c in ("-x", "x")) and (name_n == "h"):
                    col_x = pd.to_numeric(df_raw.iloc[r + 1 :, c], errors='coerce')
                    col_h = pd.to_numeric(df_raw.iloc[r + 1 :, c + 1], errors='coerce')
                    count = int((col_x.notna() & col_h.notna()).sum())
                    candidates.append((r, c, count))
        if candidates:
            r_best, c_best, _ = max(candidates, key=lambda t: t[2])
            col_x = pd.to_numeric(df_raw.iloc[r_best + 1 :, c_best], errors='coerce')
            col_h = pd.to_numeric(df_raw.iloc[r_best + 1 :, c_best + 1], errors='coerce')
            mask2 = col_x.notna() & col_h.notna()
            r = col_x[mask2].abs().to_numpy(dtype=float)
            h = col_h[mask2].to_numpy(dtype=float)
            # Align heights (bottom=0)
            try:
                if np.nanmax(h) <= 0.0 or np.nanmin(h) < 0.0:
                    h = h - float(np.nanmin(h))
            except Exception:
                pass
            order = np.argsort(h)
            return r[order], h[order]
        # If still not found, help diagnose by printing the tail of normalized headers
        headers_preview = ", ".join(_norm(c) for c in df.columns[-20:])
        raise RuntimeError("Could not find '-x' and 'h' columns in the CSV. Headers tail: " + headers_preview)

    ix_x = idx_x[-1]
    ix_h = idx_h[-1]

    r_neg = pd.to_numeric(df.iloc[:, ix_x], errors="coerce")
    h = pd.to_numeric(df.iloc[:, ix_h], errors="coerce")

    mask = r_neg.notna() & h.notna()
    r = r_neg[mask].abs().to_numpy(dtype=float)
    h = h[mask].to_numpy(dtype=float)

    # Align Excel heights to app convention: bottom=0, top=H
    try:
        if np.nanmax(h) <= 0.0 or np.nanmin(h) < 0.0:
            # Shift so minimum becomes 0 (bottom)
            h = h - float(np.nanmin(h))
    except Exception:
        pass

    # Sort by height ascending
    order = np.argsort(h)
    return r[order], h[order]


def build_app_profile(rho: float, g: float, N: float, P0: float, phi_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, h) from the app's Timoshenko solver, sorted by h ascending."""
    df, info = solve_timoshenko_membrane(rho=rho, g=g, N=N, top_pressure=P0, phi_max_deg=phi_max)
    z = df["z"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    h = float(z.max()) - z
    r = np.abs(x)
    order = np.argsort(h)
    return r[order], h[order]


def integrate_volume(r: np.ndarray, h: np.ndarray) -> float:
    """Compute V = pi * integral r(h)^2 dh (trapezoidal)."""
    return float(math.pi * np.trapz(r * r, h))


def main() -> None:
    # Adjust path if needed
    csv_path = r"C:\Users\rens_\Documents\timo.csv"

    # 1) Read Excel profile
    r_excel, h_excel = read_excel_profile(csv_path)

    # 2) Build app profile (same physics as your Excel settings)
    rho, g, N, P0, phi_max = 1000.0, 10.0, 27500.0, 100.0, 165.0
    r_app, h_app = build_app_profile(rho, g, N, P0, phi_max)

    # 3) Interpolate both on a common height grid for a fair comparison
    h_max = float(min(h_excel.max(), h_app.max()))
    h = np.linspace(0.0, h_max, 5000)
    rE = np.interp(h, h_excel, r_excel)
    rA = np.interp(h, h_app,   r_app)

    # 4) Volumes and cumulative difference
    V_excel = integrate_volume(rE, h)
    V_app   = integrate_volume(rA, h)
    # cumulative trapezoid integration (manual)
    f = (rE * rE - rA * rA)               # length n
    seg = 0.5 * (f[1:] + f[:-1]) * (h[1:] - h[:-1])  # length n-1
    dV_cum = np.concatenate([[0.0], np.cumsum(math.pi * seg)])
    dV_tot  = float(dV_cum[-1])
    idx_max = int(np.argmax(np.abs(dV_cum)))

    denom = max(V_excel, V_app)
    perc = (100.0 * dV_tot / denom) if denom > 1e-9 else float('nan')
    print(f"V_excel={V_excel:.2f} m^3  V_app={V_app:.2f} m^3  ΔV={dV_tot:.2f} m^3 ({perc:.1f}%)")
    print(f"Max ΔV_cum at h≈{h[idx_max]:.3f} m  ΔV_cum≈{dV_cum[idx_max]:.2f} m^3")

    for a, b, name in [(0.0, 0.3 * h_max, "top"), (0.3 * h_max, 0.8 * h_max, "mid"), (0.8 * h_max, h_max, "foot")]:
        sel = (h >= a) & (h < b)
        if np.any(sel):
            dV_seg = float(math.pi * np.trapz((rE[sel] * rE[sel] - rA[sel] * rA[sel]), h[sel]))
        else:
            dV_seg = 0.0
        print(f"{name:>4}: ΔV≈{dV_seg:.2f} m^3")

    # Local radius difference at the worst height
    rE_loc = float(np.interp(h[idx_max], h_excel, r_excel))
    rA_loc = float(np.interp(h[idx_max], h_app,   r_app))
    print(f"At h≈{h[idx_max]:.3f} m: r_excel={rE_loc:.3f} m, r_app={rA_loc:.3f} m")
    # Quick stats to diagnose if volumes are zero
    print(f"Excel points: {len(h_excel)}  h[min,max]=({h_excel.min():.3f},{h_excel.max():.3f})  r[max]={r_excel.max():.3f}")
    print(f" App  points: {len(h_app)}    h[min,max]=({h_app.min():.3f},{h_app.max():.3f})  r[max]={r_app.max():.3f}")


if __name__ == "__main__":
    main()


