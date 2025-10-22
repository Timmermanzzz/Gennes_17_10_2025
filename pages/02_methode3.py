"""
Methode 3: γₛ Optimalisatie (Δh Sweep)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import generate_droplet_shape, get_physical_parameters
from utils import (
    get_droplet_metrics,
    find_height_for_diameter,
    estimate_mean_curvature_at_height,
    curvature_from_head,
    delta_h_from_curvature,
    compute_torus_from_head,
)
from visualisatie import create_2d_plot
from export import export_to_stl, export_to_dxf
import tempfile

st.set_page_config(
    page_title="Methode 3 - γₛ Optimalisatie",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Methode 3 — γₛ Optimalisatie")
st.markdown("Voor gegeven afkap-diameter en kraaghoogte (Δh): vind de optimale membraanspanning (γₛ).")

# Initialize session state
if 'solutions_table_m3' not in st.session_state:
    st.session_state.solutions_table_m3 = None
if 'selected_solution_m3' not in st.session_state:
    st.session_state.selected_solution_m3 = None
if 'df_selected_m3' not in st.session_state:
    st.session_state.df_selected_m3 = None
if 'metrics_selected_m3' not in st.session_state:
    st.session_state.metrics_selected_m3 = None

st.markdown("---")

st.header("⚙️ Ontwerpparameters")

col1, col2 = st.columns(2)

with col1:
    cut_diameter_m3 = st.number_input(
        "Afkap diameter (m)",
        min_value=1.0,
        value=15.0,
        step=0.5,
        help="Gewenste opening diameter van het reservoir"
    )
    rho_m3 = st.number_input(
        "ρ - Dichtheid (kg/m³)",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

with col2:
    extra_slosh_height = st.number_input(
        "Extra kraaghoogte voor klotsen (m)",
        min_value=0.0,
        value=0.10,
        step=0.01,
        help="Extra vrije boord boven Δh"
    )
    g_m3 = st.number_input(
        "g - Zwaartekracht (m/s²)",
        min_value=0.1,
        value=9.81,
        step=0.1
    )

st.markdown("---")

st.header("🔍 Δh Bereik")

col_dh_1, col_dh_2, col_dh_3 = st.columns(3)

with col_dh_1:
    delta_h_min = st.number_input(
        "Δh min (m)",
        min_value=0.0,
        value=0.0,
        step=0.01
    )

with col_dh_2:
    delta_h_max = st.number_input(
        "Δh max (m)",
        min_value=0.01,
        value=1.0,
        step=0.01
    )

with col_dh_3:
    delta_h_step = st.number_input(
        "Δh stap (m)",
        min_value=0.001,
        value=0.01,
        step=0.001
    )

st.markdown("---")

if st.button("🔬 Genereer Oplossingen Tabel", type="primary", use_container_width=True):
    with st.spinner("Berekening..."):
        try:
            # Genereer referentie druppel met willekeurige γₛ om h_cut te vinden
            ref_gamma = 35000.0
            df_ref = generate_droplet_shape(ref_gamma, rho_m3, g_m3, cut_percentage=0)
            h_cut_ref = find_height_for_diameter(df_ref, cut_diameter_m3)
            
            if h_cut_ref is None or np.isnan(h_cut_ref):
                st.error(f"❌ Kan hoogte niet vinden voor diameter {cut_diameter_m3}m")
                st.stop()
            
            # Bepaal target kromming
            H_target = estimate_mean_curvature_at_height(df_ref, h_cut_ref)
            
            if H_target is None or np.isnan(H_target):
                st.error("❌ Kan kromming niet bepalen")
                st.stop()
            
            # Genereer tabel met verschillende Δh waarden
            delta_h_values = np.arange(float(delta_h_min), float(delta_h_max) + float(delta_h_step), float(delta_h_step))
            
            solutions = []
            
            for dh in delta_h_values:
                # Voor gegeven Δh, bereken benodigde γₛ
                # Relatie: H_target = Δp / γₛ = (ρ * g * Δh) / γₛ
                # Dus: γₛ = (ρ * g * Δh) / H_target
                
                if H_target > 0:
                    gamma_needed = (rho_m3 * g_m3 * dh) / H_target
                else:
                    gamma_needed = np.nan
                
                if np.isnan(gamma_needed) or gamma_needed <= 0:
                    continue
                
                try:
                    # Genereer druppel met deze γₛ
                    df_test = generate_droplet_shape(gamma_needed, rho_m3, g_m3, cut_percentage=0)
                    
                    # Verifieer dat we dezelfde hoogte en diameter krijgen
                    h_cut_test = find_height_for_diameter(df_test, cut_diameter_m3)
                    
                    if h_cut_test is None or np.isnan(h_cut_test):
                        continue
                    
                    # Maak afgekapte versie
                    df_cut = df_test[df_test['h'] <= h_cut_test].copy()
                    
                    if df_cut.empty:
                        continue
                    
                    # Voeg vlakke top toe
                    target_radius = cut_diameter_m3 / 2.0
                    n_points = 30
                    x_shifted_vals = np.linspace(-target_radius, target_radius, n_points)
                    x_max_current = df_cut['x-x_0'].max() if 'x-x_0' in df_cut.columns else 0.0
                    top_points_data = []
                    for x_sh in x_shifted_vals:
                        top_points_data.append({
                            'B': 1.0, 'C': 1.0, 'z': 0,
                            'x-x_0': x_sh + x_max_current,
                            'x_shifted': x_sh,
                            'h': h_cut_test
                        })
                    top_points = pd.DataFrame(top_points_data)
                    df_cut = pd.concat([df_cut, top_points], ignore_index=True)
                    
                    # Bereken metrics
                    metrics = get_droplet_metrics(df_cut)
                    
                    # Bereken torus info
                    head_total = dh + float(extra_slosh_height)
                    torus_info = compute_torus_from_head(
                        opening_diameter=cut_diameter_m3,
                        head_total=head_total,
                        wall_thickness=0.0,
                        safety_freeboard=float(extra_slosh_height)
                    )
                    
                    solutions.append({
                        'Δh (m)': round(dh, 4),
                        'γₛ (N/m)': round(gamma_needed, 0),
                        'Volume (m³)': round(metrics.get('volume', 0), 2),
                        'Max hoogte (m)': round(metrics.get('max_height', 0), 2),
                        'Basis diameter (m)': round(metrics.get('bottom_diameter', 0), 2),
                        'Max diameter (m)': round(metrics.get('max_diameter', 0), 2),
                        'Torus water (m³)': round(torus_info.get('water_volume', 0), 2),
                        '_df': df_cut,
                        '_gamma': gamma_needed,
                        '_h_cut': h_cut_test,
                    })
                
                except Exception as e:
                    continue
            
            if not solutions:
                st.error("❌ Geen geldige oplossingen gevonden. Pas parameters aan.")
                st.stop()
            
            st.session_state.solutions_table_m3 = solutions
            st.success(f"✅ {len(solutions)} oplossingen gegenereerd!")
            
        except Exception as e:
            st.error(f"❌ Fout: {str(e)}")

st.markdown("---")

# Toon tabel met oplossingen
if st.session_state.solutions_table_m3 is not None:
    st.header("📋 Oplossingen")
    
    # Verwijder interne kolommen voor display
    display_cols = ['Δh (m)', 'γₛ (N/m)', 'Volume (m³)', 'Max hoogte (m)', 
                    'Basis diameter (m)', 'Max diameter (m)', 'Torus water (m³)']
    
    solutions_display = []
    for sol in st.session_state.solutions_table_m3:
        sol_display = {col: sol[col] for col in display_cols}
        solutions_display.append(sol_display)
    
    df_solutions = pd.DataFrame(solutions_display)
    
    st.dataframe(df_solutions, use_container_width=True, height=400)
    
    st.markdown("---")
    
    st.header("🎯 Kies Oplossing")
    
    col_select_1, col_select_2 = st.columns([2, 1])
    
    with col_select_1:
        selected_idx = st.selectbox(
            "Selecteer oplossing:",
            options=range(len(st.session_state.solutions_table_m3)),
            format_func=lambda i: f"Δh = {st.session_state.solutions_table_m3[i]['Δh (m)']}m, γₛ = {st.session_state.solutions_table_m3[i]['γₛ (N/m)']:.0f} N/m",
            help="Kies de gewenste combinatie"
        )
    
    with col_select_2:
        if st.button("✅ Selecteer", use_container_width=True):
            st.session_state.selected_solution_m3 = st.session_state.solutions_table_m3[selected_idx]
            st.session_state.df_selected_m3 = st.session_state.solutions_table_m3[selected_idx]['_df']
            st.session_state.metrics_selected_m3 = {k: v for k, v in st.session_state.solutions_table_m3[selected_idx].items() 
                                                    if not k.startswith('_')}
            st.success(f"✅ Oplossing geselecteerd!")

st.markdown("---")

# Toon geselecteerde oplossing
if st.session_state.df_selected_m3 is not None:
    
    st.header("📊 Geselecteerde Oplossing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Δh (m)", f"{st.session_state.selected_solution_m3['Δh (m)']:.4f}")
        st.metric("γₛ (N/m)", f"{st.session_state.selected_solution_m3['γₛ (N/m)']:.0f}")
        st.metric("Volume (m³)", f"{st.session_state.selected_solution_m3['Volume (m³)']:.2f}")
    
    with col2:
        st.metric("Max hoogte (m)", f"{st.session_state.selected_solution_m3['Max hoogte (m)']:.2f}")
        st.metric("Basis diameter (m)", f"{st.session_state.selected_solution_m3['Basis diameter (m)']:.2f}")
        st.metric("Torus water (m³)", f"{st.session_state.selected_solution_m3['Torus water (m³)']:.2f}")
    
    st.markdown("---")
    
    st.header("📈 Visualisatie")
    fig_2d = create_2d_plot(st.session_state.df_selected_m3, metrics=st.session_state.metrics_selected_m3)
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.markdown("---")
    
    st.header("💾 Export")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        def gen_stl():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                if export_to_stl(st.session_state.df_selected_m3, tmp.name):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        stl_data = gen_stl()
        if stl_data:
            gamma_val = st.session_state.selected_solution_m3['γₛ (N/m)']
            dh_val = st.session_state.selected_solution_m3['Δh (m)']
            st.download_button(
                "📥 Download STL",
                data=stl_data,
                file_name=f"druppel_dh{dh_val:.2f}_gamma{gamma_val:.0f}.stl",
                mime="application/octet-stream",
                use_container_width=True
            )
    
    with col_exp2:
        def gen_dxf():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                if export_to_dxf(st.session_state.df_selected_m3, tmp.name):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        dxf_data = gen_dxf()
        if dxf_data:
            gamma_val = st.session_state.selected_solution_m3['γₛ (N/m)']
            dh_val = st.session_state.selected_solution_m3['Δh (m)']
            st.download_button(
                "📥 Download DXF",
                data=dxf_data,
                file_name=f"druppel_dh{dh_val:.2f}_gamma{gamma_val:.0f}.dxf",
                mime="application/dxf",
                use_container_width=True
            )

else:
    st.info("👆 Genereer oplossingen en selecteer één om te visualiseren en te exporteren.")
