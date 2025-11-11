"""
Method 2: Open Reservoir (rim-start)

Always-open reservoir solved by integrating downward from the rim.
Inputs: opening diameter, membrane tension N, density rho, gravity g.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile

from auth import require_password
from solver import solve_open_rim_membrane, shoot_open_rim_membrane
from utils import (
    get_droplet_metrics,
    init_streamlit_logger,
    get_console_logs,
    clear_console_logs,
    calculate_reservoir_surface_area,
)
from visualisatie import create_2d_plot, create_3d_plot
from export import export_to_stl, export_to_dxf


st.set_page_config(
    page_title="Method 2 - Open Reservoir (rim-start)",
    page_icon="🟦",
    layout="wide"
)

require_password()

# Console logger
logger = init_streamlit_logger(name='method2_open', level=20)

st.title("🟦 Method 2 — Open Reservoir (rim-start)")
st.markdown("Solve an always-open reservoir by starting at the opening and integrating downward.")

with st.expander("ℹ️ What is this method?", expanded=False):
    st.markdown(
        """
        - We impose the free-surface at the opening (z=0) and integrate downward.
        - Inputs: opening diameter D_open, membrane tension N, fluid density ρ and gravity g.
        - Output: total depth H under the rim, volume V, and the full profile.
        - The pressure is hydrostatic p(z) = ρ g z; the radius decreases monotonically below the rim.
        """
    )

# Keep results in session
if 'm2_df' not in st.session_state:
    st.session_state.m2_df = None
if 'm2_metrics' not in st.session_state:
    st.session_state.m2_metrics = None
if 'm2_info' not in st.session_state:
    st.session_state.m2_info = None

st.markdown("---")
st.header("⚙️ Parameters")

col_a, col_b, col_c = st.columns(3)
with col_a:
    D_open = st.number_input(
        "Opening diameter D_open (m)",
        min_value=0.1,
        max_value=1000.0,
        value=13.0,
        step=0.1,
        help="Diameter at the rim (free surface)"
    )
with col_b:
    N = st.number_input(
        "Membrane tension N (N/m)",
        min_value=100.0,
        max_value=2_000_000.0,
        value=27_500.0,
        step=500.0,
        help="Effective membrane tension per circumferential length"
    )
with col_c:
    auto_shoot = st.toggle("Auto-shoot φ_top", value=True, help="Automatically find start angle (recommended)")

col_d, col_e, col_f = st.columns(3)
with col_d:
    rho = st.number_input("ρ - Density (kg/m³)", min_value=1.0, max_value=10000.0, value=1000.0, step=50.0)
with col_e:
    g = st.number_input("g - Gravity (m/s²)", min_value=0.1, max_value=25.0, value=9.81, step=0.01)
with col_f:
    z_seed = st.number_input("z_seed – initial depth (m)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, help="Numerical kick; set 0 for pure BVP")

st.markdown("")
col_g, col_h, col_i = st.columns(3)
with col_g:
    h0 = st.number_input("Collar depth h0 (m)", min_value=0.0, max_value=2.0, value=0.00, step=0.01, help="Extra water height in rim collar; p0=ρ g h0")
with col_h:
    h_cap = st.number_input("Cap depth h_cap (m)", min_value=0.0, max_value=2.0, value=0.00, step=0.01, help="Depth below rim with constant pressure (druppelkap)")
phi_top = None
if not auto_shoot:
    phi_top = st.slider(
        "Start angle at rim φ_top (deg)",
        min_value=1.0,
        max_value=40.0,
        value=2.0,
        step=0.1,
        help="Manual start angle (use auto-shoot for physical solution)"
    )

st.markdown("")
col_action1, col_action2 = st.columns([1, 1])
with col_action1:
    if st.button("🔬 Compute Open Reservoir", type="primary", use_container_width=True):
        with st.spinner("Solving..."):
            try:
                if auto_shoot:
                    logger.info("Method2| D_open=%.3f m, N=%.1f N/m, rho=%.1f, g=%.2f, phi_top=auto-shoot", float(D_open), float(N), float(rho), float(g))
                else:
                    logger.info("Method2| D_open=%.3f m, N=%.1f N/m, rho=%.1f, g=%.2f, phi_top=%.1f deg", float(D_open), float(N), float(rho), float(g), float(phi_top or 0.0))
                if auto_shoot:
                    df, info = shoot_open_rim_membrane(
                        D_open=float(D_open), N=float(N), rho=float(rho), g=float(g),
                        h0=float(h0), h_cap=float(h_cap),
                        phi_min_deg=1.0, phi_max_deg=40.0, z_seed=float(z_seed)
                    )
                else:
                    df, info = solve_open_rim_membrane(
                        D_open=float(D_open), N=float(N), rho=float(rho), g=float(g), h0=float(h0), h_cap=float(h_cap), phi_top_deg=float(phi_top or 2.0),
                        z_seed=float(z_seed), debug=True, debug_max=80
                    )
                try:
                    logger.info("Method2| reason=%s, H=%.4f m, V=%.3f m^3, h0=%.3f m, h_cap=%.3f m, p0=%.1f Pa, steps_x=%s, steps_z=%s",
                                info.get('stopped_reason'), info.get('H_total'), info.get('volume'), float(info.get('h0', h0)), float(info.get('h_cap', h_cap)), float(info.get('p0', 0.0)),
                                info.get('steps_x', info.get('steps')), info.get('steps_z', 0))
                    # quick diagnostics
                    if df is not None and not df.empty:
                        logger.info("Method2| df: h[min,max]=[%.4f, %.4f], x[min,max]=[%.4f, %.4f]",
                                    float(df['h'].min()), float(df['h'].max()), float(df['x-x_0'].min()), float(df['x-x_0'].max()))
                        sample = df.sort_values('h').iloc[::max(1, int(len(df)/5))][['h','x-x_0']].head(5).to_dict('records')
                        logger.debug("Method2| samples(h,x)=%s", sample)
                    dbg = info.get('debug_rows')
                    if dbg:
                        for row in dbg[:10]:
                            logger.debug(
                                "dbg| %s step=%s x=%.4f z=%.4f u=%.6f dudx=%.6e dzdx=%.6e dx=%.5f -> x1=%.4f z1=%.4f u1=%.6f",
                                row.get('phase'), row.get('step'), row.get('x'), row.get('z'), row.get('u'),
                                row.get('dudx'), row.get('dzdx'), row.get('dx'), row.get('x_next'), row.get('z_next'), row.get('u_next')
                            )
                except Exception:
                    pass
                # Methode 2: eigen metrics (geen bodemschijf)
                H_total = float(info.get('H_total', float(df['h'].max()) if df is not None and not df.empty else 0.0))
                metrics = {
                    'Droplet height (m)': H_total,
                    'max_height': H_total,
                    'volume': float(info.get('volume', 0.0)),
                    'Volume (m³)': float(info.get('volume', 0.0)),
                    'top_diameter': float(D_open),
                    'Opening diameter (m)': float(D_open),
                    'bottom_diameter': 0.0,   # open reservoir: geen bodemschijf
                    'max_diameter': float(D_open),
                }
                st.session_state.m2_df = df
                st.session_state.m2_metrics = metrics
                st.session_state.m2_info = info
                st.success("✅ Computation successful!")
            except Exception as e:
                logger.exception("Method2 error: %s", str(e))
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")

if st.session_state.m2_df is not None:
    st.header("📊 Specifications")
    m = st.session_state.m2_metrics or {}
    info = st.session_state.m2_info or {}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Droplet height (m)", f"{float(info.get('H_total', 0.0)):.2f}")
        st.metric("Volume (m³)", f"{float(info.get('volume', 0.0)):.2f}")
    with col2:
        st.metric("Opening diameter (m)", f"{float(m.get('top_diameter', 0.0)):.2f}")
        st.metric("Max diameter (m)", f"{float(m.get('max_diameter', 0.0)):.2f}")
    with col3:
        base_d = float(m.get('bottom_diameter', 0.0) or 0.0)
        st.metric("Base diameter (m)", f"{base_d:.2f}")

    st.markdown("")
    st.subheader("🧵 Skin surface")
    try:
        area_res = float(calculate_reservoir_surface_area(st.session_state.m2_df, include_bottom_disc=True))
    except Exception:
        area_res = 0.0
    st.metric("Reservoir skin (m²)", f"{area_res:.2f}")

    st.markdown("---")
    st.header("📈 Visualisation")
    st.subheader("2D Cross-section")
    # Fix B: ignore any precomputed x_shifted and let the plotter place rim at 0 using 'x-x_0'
    df_plot = st.session_state.m2_df.copy()
    try:
        if 'x_shifted' in df_plot.columns:
            df_plot = df_plot.drop(columns=['x_shifted'])
    except Exception:
        pass
    fig2d = create_2d_plot(df_plot, metrics=st.session_state.m2_metrics, title="Open Reservoir Profile", view="full", show_seam=False, show_cut_plane=False)
    try:
        H = float(st.session_state.m2_info.get('H_total', st.session_state.m2_df['h'].max()))
        R = float(st.session_state.m2_info.get('R_open', abs(st.session_state.m2_df['x-x_0']).max()))
        # Zoom op werkelijk profiel en gelijke schaal
        fig2d.update_yaxes(range=[-0.05 * H, 1.05 * H])
        fig2d.update_xaxes(range=[-1.05 * R, 0.05 * R], scaleanchor="y", scaleratio=1)
    except Exception:
        pass
    st.plotly_chart(fig2d, use_container_width=True)

    st.subheader("3D Model")
    fig3d = create_3d_plot(st.session_state.m2_df, metrics=st.session_state.m2_metrics, title="Open Reservoir 3D")
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("---")
    st.header("💾 Export")
    ce1, ce2 = st.columns(2)
    with ce1:
        def _gen_stl():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                if export_to_stl(st.session_state.m2_df, tmp.name, metrics=st.session_state.m2_metrics):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        stl_data = _gen_stl()
        if stl_data:
            st.download_button("📥 Download STL", data=stl_data, file_name="open_reservoir.stl", mime="application/octet-stream", use_container_width=True)
    with ce2:
        def _gen_dxf():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                if export_to_dxf(st.session_state.m2_df, tmp.name, metrics=st.session_state.m2_metrics):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        dxf_data = _gen_dxf()
        if dxf_data:
            st.download_button("📥 Download DXF", data=dxf_data, file_name="open_reservoir.dxf", mime="application/dxf", use_container_width=True)

st.markdown("---")
st.header("🖥️ Console")
col_log1, col_log2 = st.columns([1, 0.2])
with col_log2:
    if st.button("Clear logs", use_container_width=True):
        clear_console_logs()
logs = "\n".join(get_console_logs())
st.code(logs or "(no logs yet)")


