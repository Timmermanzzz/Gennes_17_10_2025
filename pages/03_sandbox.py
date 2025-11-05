"""
Sandbox: interactively explore the uncut Timoshenko droplet shape.

Sliders: YS (N/m), density, gravity, top pressure (or P0 = rho*g*h_w), phi_max.
We only compute and show the full (uncut) profile and a few key metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np

from auth import require_password
from solver import solve_timoshenko_membrane
from visualisatie import create_2d_plot


st.set_page_config(page_title="Sandbox – Timoshenko (uncut)", page_icon="🧪", layout="wide")
require_password()

st.title("🧪 Sandbox — Timoshenko (uncut shape)")
st.caption("Interactively explore how YS, density, gravity and top pressure (P₀) influence the uncut droplet shape.")

st.markdown("---")

col_a, col_b, col_c = st.columns(3)
with col_a:
    ys = st.slider("YS – Membrane tension (N/m)", min_value=0, max_value=100_000, value=27_500, step=500)
    rho = st.slider("ρ – Density (kg/m³)", min_value=500, max_value=2_000, value=1_000, step=10)
with col_b:
    g = st.slider("g – Gravity (m/s²)", min_value=1.0, max_value=20.0, value=9.81, step=0.01)
    phi_max = st.slider("φ max (deg)", min_value=60, max_value=170, value=120, step=5)
with col_c:
    p0 = st.slider("P₀ – Top pressure (Pa)", min_value=0, max_value=2_000, value=0, step=10)

st.markdown("P₀ = {:.1f} Pa".format(p0))

# Physically closed shape option
closed_shape = st.toggle("Closed shape (stop at x→0)", value=True, help="Integrate until the meridian closes (x → 0). Overrides φ max.")

st.markdown("---")

err = None
df_vis = None
info = {}
try:
    phi_eff = 179.0 if closed_shape else float(phi_max)
    df, info = solve_timoshenko_membrane(rho=float(rho), g=float(g), N=float(ys), top_pressure=float(p0), phi_max_deg=float(phi_eff))
    # Convert to our plotting convention: h upward from bottom; x_shifted negative to the left
    z_arr = np.asarray(df['z'], dtype=float)
    x_arr = np.asarray(df['x'], dtype=float)
    H_max = float(np.max(z_arr)) if len(z_arr) else 0.0
    h_arr = H_max - z_arr
    df_vis = pd.DataFrame({'h': h_arr, 'x-x_0': x_arr})
    try:
        df_vis['x_shifted'] = -pd.Series(x_arr, dtype=float)
    except Exception:
        pass
except Exception as e:
    err = str(e)

col1, col2 = st.columns([2, 1])
with col1:
    if err:
        st.error(f"Solver error: {err}")
    elif df_vis is None or df_vis.empty:
        st.warning("No data to show.")
    else:
        st.subheader("2D cross-section (uncut)")
        fig_placeholder = st.empty()
        fig = create_2d_plot(df_vis, metrics=None, title="Uncut profile", view="full", show_seam=False, show_cut_plane=False)

        st.markdown("---")
        st.subheader("View settings")
        lock_axes = st.toggle("Lock 2D scale (fixed axes)", value=True)
        x_min, x_max, y_min, y_max = -20.0, 2.0, -1.0, 8.0
        if lock_axes:
            c1, c2 = st.columns(2)
            with c1:
                x_min = st.slider("x min (m)", min_value=-50.0, max_value=0.0, value=-20.0, step=0.5)
                x_max = st.slider("x max (m)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            with c2:
                y_min = st.slider("y min (m)", min_value=-2.0, max_value=10.0, value=-1.0, step=0.5)
                y_max = st.slider("y max (m)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
            fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(range=[y_min, y_max])

        fig_placeholder.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Key metrics")
    if df_vis is not None and not df_vis.empty:
        x = df_vis['x-x_0'].to_numpy(dtype=float)
        z = H_max - df_vis['h'].to_numpy(dtype=float)  # back to z
        # report full height/diameters
        H_full = float(H_max)
        D_max = float(2.0 * np.max(np.abs(x))) if len(x) else 0.0
        st.metric("Full height H (m)", f"{H_full:.2f}")
        st.metric("Max diameter (m)", f"{D_max:.2f}")
        st.metric("r₁ apex (m)", f"{float(info.get('r1_apex', 0.0)):.2f}")
        st.metric("head d (m)", f"{float(info.get('head_d', 0.0)):.3f}")
        st.caption("These are metrics of the uncut shape.")


