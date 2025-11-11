"""
Diagnostic runner for the open-rim solver (Method 2).

Usage:
  python test_open_rim.py

It sweeps over start angles and seed depths, prints:
  - Reason, H, V, steps_x/z
  - u at switch (last x-phase row), expected neutral du/dx sign at start
  - Monotonicity checks on returned profile (h up, x down)
  - Quick assessment if early switch is likely
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from solver import solve_open_rim_membrane


def neutral_phi_top_deg(D_open: float, N: float, rho: float, g: float, z_seed: float) -> float:
    """Compute neutral start angle where u/x ≈ (rho*g*z_seed)/N.
    Returns phi_top in degrees.
    """
    R = float(D_open) / 2.0
    term = (float(rho) * float(g) * float(z_seed)) / float(N)
    u = max(1e-9, term * R)
    u = min(u, 0.999999)
    return float(math.degrees(math.asin(u)))


def check_monotonicity(df: pd.DataFrame) -> tuple[bool, bool]:
    """Return (h_increasing, x_decreasing)."""
    if df is None or df.empty:
        return False, False
    d = df.sort_values('h')
    h = d['h'].to_numpy(dtype=float)
    x = d['x-x_0'].to_numpy(dtype=float)
    return bool(np.all(np.diff(h) >= -1e-9)), bool(np.all(np.diff(x) <= 1e-9))


def run_case(D_open: float, N: float, rho: float, g: float, phi_top: float, z_seed: float) -> None:
    print("\n=== CASE D=%.2f N=%.1f rho=%.1f g=%.2f phi_top=%.3f z_seed=%.3f ===" % (D_open, N, rho, g, phi_top, z_seed))
    df, info = solve_open_rim_membrane(
        D_open=float(D_open), N=float(N), rho=float(rho), g=float(g),
        phi_top_deg=float(phi_top), z_seed=float(z_seed), debug=True, debug_max=200
    )
    reason = info.get('stopped_reason')
    H = float(info.get('H_total', 0.0))
    V = float(info.get('volume', 0.0))
    sx = int(info.get('steps_x', info.get('steps', 0)))
    sz = int(info.get('steps_z', 0))
    print("reason=%s, H=%.4f m, V=%.3f m^3, steps_x=%d, steps_z=%d" % (reason, H, V, sx, sz))

    dbg = info.get('debug_rows') or []
    if dbg:
        first = dbg[0]
        last = dbg[-1]
        print("x-phase start:   x=%.4f z=%.4f u=%.6f dudx=%.3e dzdx=%.3e dx=%.5f" % (first['x'], first['z'], first['u'], first['dudx'], first['dzdx'], first['dx']))
        print("x-phase at end:  x=%.4f z=%.4f u=%.6f dudx=%.3e dzdx=%.3e dx=%.5f" % (last['x'], last['z'], last['u'], last['dudx'], last['dzdx'], last['dx']))
        u_switch = float(last['u'])
    else:
        u_switch = float('nan')
    print("u at switch (est from x-phase last) = %.6f" % u_switch)

    # Monotonicity
    h_inc, x_dec = check_monotonicity(df)
    print("monotonicity: h_increasing=%s, x_decreasing=%s" % (h_inc, x_dec))

    # Early switch heuristic
    R = D_open / 2.0
    likely_early = (H < 1.0) or (u_switch < 0.7) or (sz == 0)
    print("assessment: likely_early_switch=%s (H<1 or u_switch<0.7 or steps_z==0)" % likely_early)


def main() -> None:
    D_open = 13.0
    N = 27500.0
    rho = 1000.0
    g = 9.81

    z_seeds = [0.02, 0.05, 0.10]
    for z_seed in z_seeds:
        phi_neutral = neutral_phi_top_deg(D_open, N, rho, g, z_seed)
        phi_list = [max(1.0, phi_neutral - 0.8), phi_neutral, phi_neutral + 0.8]
        print("\n=== z_seed=%.3f, phi_neutral≈%.3f° ===" % (z_seed, phi_neutral))
        for phi_top in phi_list:
            run_case(D_open, N, rho, g, phi_top, z_seed)


if __name__ == "__main__":
    main()


