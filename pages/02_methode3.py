"""
Method 3: Volume-Based Collar Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import generate_droplet_shape, get_physical_parameters
from utils import (
    get_droplet_metrics,
    find_height_for_diameter,
    calculate_diameter_at_height,
    find_collar_tube_diameter_for_volume,
    solve_gamma_for_cut_volume_match
)
from visualisatie import create_2d_plot, create_3d_plot
from export import export_to_stl, export_to_dxf
import tempfile
from io import BytesIO

st.set_page_config(
    page_title="Method 3 - γₛ Optimization",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Method 3 — Volume-Based Collar Design")
st.markdown("Find collar tube diameters where collar volume equals cut volume.")

# Uitleg/Help
with st.expander("ℹ️ Help — How does Method 3 work?", expanded=False):
    st.markdown(
        """
        - **Goal**: Design a collar for a chosen opening so that **Collar volume = Cut volume**.
        - **Exact physics**: Collar tube diameter D is solved from
          \(\;\text{Cut} = A·(D − s) − \pi^2 R (D/2)^2\;\) (s = sloshing, here default 0).
        - **Inputs**: Cut diameter (m), γₛ (N/m), ρ (kg/m³), g (m/s²).
        - **Workflow**:
          1) Compute the cut height and cut volume.
          2) Solve D with displacement-aware formula.
          3) Show the volume match and geometry; select and export.
        - **Units**: m and m³.
        """
    )

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

st.header("⚙️ Design parameters")

col1, col2 = st.columns(2)

with col1:
    cut_diameter_m3 = st.number_input(
        "Cut diameter (m)",
        min_value=1.0,
        value=15.0,
        step=0.5,
        help="Desired opening diameter of the reservoir"
    )
    gamma_s_m3 = st.number_input(
        "γₛ - Surface tension (N/m)",
        min_value=100.0,
        max_value=1000000.0,
        value=35000.0,
        step=1000.0
    )

with col2:
    rho_m3 = st.number_input(
        "ρ - Density (kg/m³)",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )
    g_m3 = st.number_input(
        "g - Gravity (m/s²)",
        min_value=0.1,
        value=9.81,
        step=0.1
    )

st.markdown("---")

st.header("🔍 Tube Diameter Range")

col_td_1, col_td_2, col_td_3 = st.columns(3)

with col_td_1:
    tube_diameter_min = st.number_input(
        "Tube diameter min (m)",
        min_value=0.01,
        value=0.10,
        step=0.01
    )

with col_td_2:
    tube_diameter_max = st.number_input(
        "Tube diameter max (m)",
        min_value=0.01,
        value=0.50,
        step=0.01
    )

with col_td_3:
    tube_diameter_step = st.number_input(
        "Tube diameter step (m)",
        min_value=0.001,
        value=0.01,
        step=0.001
    )

st.markdown("---")

st.subheader("🌊 Collar options")
sloshing_m3 = st.number_input(
    "Sloshing height (m)",
    min_value=0.0,
    value=0.0,
    step=0.01,
    help="Extra vrije boord boven het water in de kraag"
)

if st.button("🔬 Generate Solutions Table", type="primary", use_container_width=True):
    with st.spinner("Computing..."):
        try:
            # Genereer druppel met opgegeven γₛ
            df_full = generate_droplet_shape(gamma_s_m3, rho_m3, g_m3, cut_percentage=0)
            h_cut = find_height_for_diameter(df_full, cut_diameter_m3)
            
            if h_cut is None or np.isnan(h_cut):
                st.error(f"❌ Cannot find height for diameter {cut_diameter_m3} m")
                st.stop()
            
            # Maak afgekapte versie
            df_cut_raw = df_full[df_full['h'] <= h_cut].copy()
            
            if df_cut_raw.empty:
                st.error("❌ Cannot create cut shape")
                st.stop()
            
            # Bereken volume ZONDER vlakke top
            metrics_cut_raw = get_droplet_metrics(df_cut_raw)
            volume_cut_raw = metrics_cut_raw.get('volume', 0.0)
            
            # Voeg vlakke top toe
            target_radius = cut_diameter_m3 / 2.0
            n_points = 30
            x_shifted_vals = np.linspace(0.0, target_radius, n_points)
            top_points_data = []
            x_max_current = df_cut_raw['x-x_0'].max() if 'x-x_0' in df_cut_raw.columns else 0.0
            for x_sh in x_shifted_vals:
                top_points_data.append({
                    'B': 1.0, 'C': 1.0, 'z': 0,
                    'x-x_0': x_sh + x_max_current,
                    'x_shifted': x_sh,
                    'h': h_cut
                })
            top_points = pd.DataFrame(top_points_data)
            subset_cols = ['x-x_0', 'h'] if 'x-x_0' in df_cut_raw.columns else ['x_shifted', 'h']
            df_cut = pd.concat([df_cut_raw, top_points], ignore_index=True).drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
            
            # Bereken volume_full en cut_volume
            metrics_full = get_droplet_metrics(df_full)
            volume_full = metrics_full.get('volume', 0.0)
            cut_volume = volume_full - volume_cut_raw
            
            # Sweep over tube diameters
            tube_values = np.arange(tube_diameter_min, tube_diameter_max + 0.5 * tube_diameter_step, tube_diameter_step)
            solutions = []
            for tube_diam in tube_values:
                R_major = cut_diameter_m3 / 2.0
                r_tube = tube_diam / 2.0
                # Netto watervolume B = A*(D - s) - π² R (D/2)²
                water_height = max(0.0, tube_diam - float(sloshing_m3))
                collar_vol_phys = (np.pi * (R_major ** 2)) * water_height - ((np.pi ** 2) * R_major * (r_tube ** 2))
                # Find gamma_s such that cut volume equals collar_vol_phys
                gamma_match, df_cut_gamma, cut_vol_gamma = solve_gamma_for_cut_volume_match(
                    target_cut_volume=float(collar_vol_phys),
                    rho=float(rho_m3),
                    g=float(g_m3),
                    cut_diameter=float(cut_diameter_m3),
                    gamma_min=100.0,
                    gamma_max=1_000_000.0,
                    max_iter=25,
                    rel_tol=1e-3,
                )
                metrics = get_droplet_metrics(df_cut_gamma)
                entry = {
                    'Tube diameter (m)': round(float(tube_diam), 3),
                    'Collar volume (m³)': round(float(collar_vol_phys), 2),
                    'Cut volume (m³)': round(float(cut_vol_gamma), 2),
                    'Volume match (%)': round(100.0 * (cut_vol_gamma / max(collar_vol_phys, 1e-9)), 1),
                    'γₛ match (N/m)': round(float(gamma_match), 1),
                    'Reservoir volume (m³)': round(metrics.get('volume', 0), 2),
                    'Max height (m)': round(metrics.get('max_height', 0), 2),
                    'Base diameter (m)': round(metrics.get('bottom_diameter', 0), 2),
                    'Max diameter (m)': round(metrics.get('max_diameter', 0), 2),
                    # Intern voor visualisatie/export van de selectie
                    'torus_R_major': R_major,
                    'torus_r_top': r_tube,
                    'torus_r_water': r_tube,
                    'torus_water_volume': collar_vol_phys,
                    'h_seam_eff': float(h_cut),
                    'collar_tube_diameter': float(tube_diam),
                    '_df': df_cut_gamma,
                    '_gamma': float(gamma_match),
                    '_h_cut': h_cut,
                    'volume_afgekapt': float(cut_vol_gamma),
                    'volume_kraag': float(collar_vol_phys),
                }
                # Waterline at seam + (tube height - sloshing)
                entry['h_waterline'] = float(h_cut + water_height)
                solutions.append(entry)
            
            if not solutions:
                st.error("❌ No solutions generated.")
                st.stop()
            
            st.session_state.solutions_table_m3 = solutions
            st.success(f"✅ Generated {len(solutions)} solutions!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")

# Show solutions table
if st.session_state.solutions_table_m3 is not None:
    st.header("📋 Solutions")
    
    # Verwijder interne kolommen voor display
    display_cols = ['Tube diameter (m)', 'Collar volume (m³)', 'Cut volume (m³)', 'Volume match (%)',
                    'γₛ match (N/m)', 'Reservoir volume (m³)', 'Max height (m)', 'Base diameter (m)', 'Max diameter (m)']
    
    solutions_display = []
    for sol in st.session_state.solutions_table_m3:
        sol_display = {col: sol[col] for col in display_cols}
        solutions_display.append(sol_display)
    
    df_solutions = pd.DataFrame(solutions_display)
    
    st.dataframe(df_solutions, use_container_width=True, height=400)

    # Export buttons
    col_exp_tbl_1, col_exp_tbl_2 = st.columns(2)
    with col_exp_tbl_1:
        csv_bytes = df_solutions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download table (CSV)",
            data=csv_bytes,
            file_name="methode3_oplossingen.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp_tbl_2:
        try:
            bio = BytesIO()
            with pd.ExcelWriter(bio) as writer:
                df_solutions.to_excel(writer, index=False, sheet_name='solutions')
            xlsx_data = bio.getvalue()
            st.download_button(
                label="📥 Download table (Excel)",
                data=xlsx_data,
                file_name="methode3_oplossingen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as _:
            st.info("Excel export not available (missing package). Use CSV.")
    
    st.markdown("---")
    
    st.header("🎯 Choose Solution")
    
    col_select_1, col_select_2 = st.columns([2, 1])
    
    with col_select_1:
        selected_idx = st.selectbox(
            "Select solution:",
            options=range(len(st.session_state.solutions_table_m3)),
            format_func=lambda i: f"Tube = {st.session_state.solutions_table_m3[i]['Tube diameter (m)']:.2f}m, Match = {st.session_state.solutions_table_m3[i]['Volume match (%)']:.1f}%",
            help="Choose the desired combination"
        )
    
    with col_select_2:
        if st.button("✅ Select", use_container_width=True):
            st.session_state.selected_solution_m3 = st.session_state.solutions_table_m3[selected_idx]
            st.session_state.df_selected_m3 = st.session_state.solutions_table_m3[selected_idx]['_df']
            st.session_state.metrics_selected_m3 = {k: v for k, v in st.session_state.solutions_table_m3[selected_idx].items() 
                                                    if not k.startswith('_')}
            st.success(f"✅ Solution selected!")

st.markdown("---")

# Show selected solution
if st.session_state.df_selected_m3 is not None:
    
    st.header("📊 Selected Solution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Tube diameter (m)", f"{st.session_state.selected_solution_m3['Tube diameter (m)']:.3f}")
        st.metric("Collar volume (m³)", f"{st.session_state.selected_solution_m3['Collar volume (m³)']:.2f}")
        st.metric("Cut volume (m³)", f"{st.session_state.selected_solution_m3['Cut volume (m³)']:.2f}")
        st.metric("Volume match (%)", f"{st.session_state.selected_solution_m3['Volume match (%)']:.1f}")
    
    with col2:
        st.metric("Reservoir volume (m³)", f"{st.session_state.selected_solution_m3['Reservoir volume (m³)']:.2f}")
        st.metric("Max height (m)", f"{st.session_state.selected_solution_m3['Max height (m)']:.2f}")
        st.metric("Base diameter (m)", f"{st.session_state.selected_solution_m3['Base diameter (m)']:.2f}")
        st.metric("Max diameter (m)", f"{st.session_state.selected_solution_m3['Max diameter (m)']:.2f}")
    
    st.markdown("---")
    
    st.header("📈 Visualisation")
    st.subheader("2D Cross-section")
    fig_2d = create_2d_plot(
        st.session_state.df_selected_m3,
        metrics=st.session_state.metrics_selected_m3,
        view="full",
        show_seam=False,
        show_cut_plane=True,
        cut_plane_h=st.session_state.selected_solution_m3.get('_h_cut', None)
    )
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.subheader("3D Model")
    fig_3d = create_3d_plot(
        st.session_state.df_selected_m3,
        metrics=st.session_state.metrics_selected_m3,
        title="Droplet 3D Model"
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
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
            tube_val = st.session_state.selected_solution_m3['Tube diameter (m)']
            match_val = st.session_state.selected_solution_m3['Volume match (%)']
            st.download_button(
                "📥 Download STL",
                data=stl_data,
                file_name=f"droplet_tube{tube_val:.2f}m_match{match_val:.0f}%.stl",
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
            tube_val = st.session_state.selected_solution_m3['Tube diameter (m)']
            match_val = st.session_state.selected_solution_m3['Volume match (%)']
            st.download_button(
                "📥 Download DXF",
                data=dxf_data,
                file_name=f"droplet_tube{tube_val:.2f}m_match{match_val:.0f}%.dxf",
                mime="application/dxf",
                use_container_width=True
            )

else:
    st.info("👆 Generate solutions and select one to visualise and export.")
