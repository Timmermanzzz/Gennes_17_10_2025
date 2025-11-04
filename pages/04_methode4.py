"""
Method 4: Variationale druppelvorm (energie-minimalisatie bij vaste snijhoogte)
"""

import streamlit as st
import pandas as pd
import numpy as np
from solver import solve_variational_droplet_fixed_cut, get_physical_parameters, generate_droplet_shape
from utils import get_droplet_metrics, shift_x_coordinates, find_height_for_diameter
from visualisatie import create_2d_plot, create_3d_plot
from export import export_to_stl, export_to_dxf, export_to_3dm
from pdf_export import export_to_pdf
from auth import require_password
import tempfile

st.set_page_config(
    page_title="Method 4 - Variational Energy",
    page_icon="🧮",
    layout="wide"
)

require_password()

st.title("🧮 Method 4 — Variationale druppelvorm (vaste snijhoogte)")
st.markdown("Minimaliseer totale energie met vaste snijhoogte en openingdiameter. Volume volgt uit de oplossing.")

with st.expander("ℹ️ Help — Variationale methode", expanded=False):
    st.markdown(
        """
        - Doel: vind r(z) die de energie E = γₛ·A + ρg·∫ z dV minimaliseert, met randvoorwaarden r(0)=0 en r(h_cut)=D_open/2.
        - Invoer: γₛ (N/m), ρ (kg/m³), g (m/s²), openingdiameter D_open (m), snijhoogte h_cut (m).
        - Uitvoer: profiel/mesh, volume, oppervlaktte, basis-/maxdiameter, hoogte.
        - Gebruik: robuuste alternatief/validator naast de Laplace-gebaseerde methode.
        """
    )

# Session state init
if 'm4_df' not in st.session_state:
    st.session_state.m4_df = None
if 'm4_metrics' not in st.session_state:
    st.session_state.m4_metrics = None
if 'm4_phys' not in st.session_state:
    st.session_state.m4_phys = None

st.markdown("---")
st.header("⚙️ Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    gamma_s = st.number_input(
        "γₛ - Surface tension (N/m)",
        min_value=100.0,
        max_value=1000000.0,
        value=27500.0,
        step=500.0,
        help="Oppervlaktespanning (instelbaar)"
    )
with col2:
    rho = st.number_input(
        "ρ - Density (kg/m³)",
        min_value=1.0,
        max_value=10000.0,
        value=1000.0,
        step=50.0,
        help="Dichtheid van de vloeistof"
    )
with col3:
    g = st.number_input(
        "g - Gravity (m/s²)",
        min_value=0.1,
        max_value=20.0,
        value=9.8,
        step=0.1,
        help="Gravitatieversnelling"
    )

col4, col5, col6 = st.columns(3)
with col4:
    opening_diameter = st.number_input(
        "Opening diameter (m)",
        min_value=0.1,
        max_value=200.0,
        value=13.0,
        step=0.1,
        help="Diameter van de opening (bovenkant)"
    )
with col5:
    cut_height = st.number_input(
        "Snijhoogte h_cut (m)",
        min_value=0.05,
        max_value=200.0,
        value=3.0,
        step=0.05,
        help="Hoogte waarop de druppel wordt afgesneden"
    )
with col6:
    num_points = st.slider("Resolutie (punten)", min_value=100, max_value=1200, value=400, step=50)

with st.expander("Geavanceerd", expanded=False):
    col_adv1, col_adv2, col_adv3 = st.columns(3)
    with col_adv1:
        reg_w = st.number_input("Regularisatie gewicht", min_value=0.0, max_value=1e-2, value=1e-6, step=1e-6, format="%e")
    with col_adv2:
        max_iter = st.number_input("Max iteraties", min_value=50, max_value=5000, value=500, step=50)
    with col_adv3:
        tol = st.number_input("Tolerantie (ftol)", min_value=1e-12, max_value=1e-3, value=1e-6, step=1e-6, format="%e")

st.markdown("")
colA, colB = st.columns([1, 1])
with colA:
    if st.button("🔬 Bereken variationale vorm", type="primary", use_container_width=True):
        with st.spinner("Optimaliseren..."):
            try:
                df, info = solve_variational_droplet_fixed_cut(
                    gamma_s=gamma_s,
                    rho=rho,
                    g=g,
                    opening_diameter=opening_diameter,
                    cut_height=cut_height,
                    num_points=int(num_points),
                    regularization_weight=float(reg_w),
                    max_iter=int(max_iter),
                    tol=float(tol)
                )
                # Zorg voor x_shifted
                if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
                    df = df.copy()
                    x_max = df['x-x_0'].max()
                    df['x_shifted'] = df['x-x_0'] - x_max

                metrics = get_droplet_metrics(df)
                # Overschrijf/forceer vaste opening/snede
                metrics['h_cut'] = float(cut_height)
                metrics['top_diameter'] = float(opening_diameter)
                metrics['max_height'] = float(cut_height)

                phys = get_physical_parameters(df, gamma_s, rho, g)
                phys['gamma_s'] = gamma_s
                phys['rho'] = rho
                phys['g'] = g

                st.session_state.m4_df = df
                st.session_state.m4_metrics = metrics
                st.session_state.m4_phys = phys
                st.success(f"✅ Variationale vorm berekend. Volume ≈ {metrics.get('volume', 0):.2f} m³")
            except Exception as e:
                st.error(f"❌ Fout: {e}")

st.markdown("---")

if st.session_state.m4_df is not None:
    st.header("📊 Specificaties")
    m = st.session_state.m4_metrics or {}
    p = st.session_state.m4_phys or {}

    colv1, colv2, colv3 = st.columns(3)
    with colv1:
        st.metric("Droplet Volume (m³)", f"{m.get('volume', 0):.2f}")
    with colv2:
        st.metric("Droplet height (m)", f"{m.get('max_height', 0):.2f}")
    with colv3:
        st.metric("Opening diameter (m)", f"{m.get('top_diameter', 0):.2f}")

    colg1, colg2 = st.columns(2)
    with colg1:
        st.metric("Max diameter (m)", f"{m.get('max_diameter', 0):.2f}")
    with colg2:
        st.metric("Base diameter (m)", f"{m.get('bottom_diameter', 0):.2f}")

    st.markdown("---")
    st.header("📈 Visualisatie")
    st.subheader("2D doorsnede")
    fig2d = create_2d_plot(
        st.session_state.m4_df,
        metrics={ 'h_cut': m.get('h_cut', cut_height), 'top_diameter': m.get('top_diameter', opening_diameter) },
        view="full",
        show_seam=False,
        show_cut_plane=True,
        cut_plane_h=m.get('h_cut', cut_height)
    )
    st.plotly_chart(fig2d, use_container_width=True)

    st.subheader("3D Model")
    fig3d = create_3d_plot(st.session_state.m4_df, metrics={ 'h_cut': m.get('h_cut', cut_height) })
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("---")
    st.header("💾 Export")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        def gen_stl():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                if export_to_stl(st.session_state.m4_df, tmp.name, metrics=st.session_state.m4_metrics):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        stl_data = gen_stl()
        if stl_data:
            st.download_button("📥 Download STL", data=stl_data, file_name="droplet_m4.stl", mime="application/octet-stream", use_container_width=True)

    with c2:
        def gen_dxf():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                if export_to_dxf(st.session_state.m4_df, tmp.name, metrics=st.session_state.m4_metrics):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        dxf_data = gen_dxf()
        if dxf_data:
            st.download_button("📥 Download DXF", data=dxf_data, file_name="droplet_m4.dxf", mime="application/dxf", use_container_width=True)

    with c3:
        if st.button("📄 Generate PDF (A3)", use_container_width=True):
            pdf_bytes = None
            err = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    export_to_pdf(st.session_state.m4_df, st.session_state.m4_metrics, tmp.name, physical_params=st.session_state.m4_phys)
                    with open(tmp.name, 'rb') as f:
                        pdf_bytes = f.read()
            except Exception as e:
                err = str(e)
            if pdf_bytes:
                st.download_button("📥 Download PDF (A3)", data=pdf_bytes, file_name="droplet_m4_a3.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.error(f"PDF genereren mislukt. {err or ''}")

    with c4:
        def gen_3dm():
            import os
            fd, tmp_path = tempfile.mkstemp(suffix='.3dm')
            os.close(fd)
            ok, err = export_to_3dm(st.session_state.m4_df, tmp_path, metrics=st.session_state.m4_metrics)
            if ok and os.path.exists(tmp_path):
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return data, None
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return None, err
        if st.button("🦏 Download 3DM", use_container_width=True):
            data3dm, err = gen_3dm()
            if data3dm:
                st.download_button("📥 3DM", data=data3dm, file_name="droplet_m4.3dm", mime="application/octet-stream", use_container_width=True)
            else:
                st.error(f"3DM export mislukt. {err or 'Controleer of rhino3dm is geïnstalleerd.'}")

    st.markdown("---")
    with st.expander("🔍 Vergelijk met Methode 1 (Young–Laplace)", expanded=False):
        colc1, colc2 = st.columns([1, 2])
        with colc1:
            if st.button("Vergelijk nu", use_container_width=True):
                try:
                    df_full = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                    h_cut_m1 = find_height_for_diameter(df_full, float(opening_diameter))
                    if np.isnan(h_cut_m1):
                        st.warning("Methode 1: geen snijhoogte gevonden voor deze opening.")
                    else:
                        df_m1 = df_full[df_full['h'] <= h_cut_m1].copy()
                        # Voeg vlakke top toe rechts (0..R)
                        R_top = float(opening_diameter) / 2.0
                        n_top = 30
                        x_sh_vals = np.linspace(-R_top, 0.0, n_top)
                        x_max_current = df_m1['x-x_0'].max() if 'x-x_0' in df_m1.columns else 0.0
                        top_points = pd.DataFrame({
                            'B': 1.0,
                            'C': 1.0,
                            'z': 0.0,
                            'x-x_0': x_sh_vals + x_max_current,
                            'x_shifted': x_sh_vals,
                            'h': float(h_cut_m1)
                        })
                        df_m1 = pd.concat([df_m1, top_points], ignore_index=True)
                        m_m1 = get_droplet_metrics(df_m1)

                        # Metrics-vergelijking
                        vol_m4 = float(m.get('volume', 0))
                        vol_m1 = float(m_m1.get('volume', 0))
                        dvol_pct = 0.0 if vol_m1 == 0 else 100.0 * (vol_m4 - vol_m1) / vol_m1
                        st.metric("Δ Volume (%)", f"{dvol_pct:+.2f}%", help="(M4 - M1) / M1 · 100%")

                        # Toon 2D-vergelijkingsplotten naast elkaar
                        fig_m1 = create_2d_plot(
                            df_m1,
                            metrics={'h_cut': float(h_cut_m1), 'top_diameter': float(opening_diameter)},
                            view="full",
                            show_cut_plane=True,
                            cut_plane_h=float(h_cut_m1)
                        )
                        st.plotly_chart(fig_m1, use_container_width=True)
                except Exception as e:
                    st.error(f"Vergelijking mislukt: {e}")

else:
    st.info("👆 Vul parameters in en klik op ‘Bereken variationale vorm’.")


