"""
Method 3: γₛ Optimization (Δh sweep)
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
    compute_collar_segment_volume,
    find_delta_h_for_collar_volume,
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

st.title("🎯 Method 3 — γₛ Optimization")
st.markdown("For a given cut diameter and collar head (Δh): find the optimal membrane tension (γₛ).")

# Uitleg/Help
with st.expander("ℹ️ Help — How does Method 3 work?", expanded=False):
    st.markdown(
        """
        - **Goal**: Find combinations of **Δh** (collar head) and **γₛ** (membrane tension) that restore the **closed reference curvature** at a fixed **cut diameter**.
        - **What is γₛ?** Effective **membrane tension** (N/m). Not stiffness (E) or pressure; with **Δp = 2 γₛ H** it sets curvature. Higher γₛ ⇒ flatter; lower γₛ ⇒ rounder.
        - **Compensation principle** (as in Method 1):
          - The **rigid ring** fixes the opening (diameter) after cutting.
          - The **collar** water provides the missing **pressure head**: extra pressure **ρ g Δh** at the edge.
          - With **γₛ** we set how ‘tight’ the membrane is; together with pressure this sets the **curvature** (Young–Laplace).
        - **Workflow**:
          1. You define the **Δh range** (and step); we compute for each Δh the **γₛ** that yields the reference curvature.
          2. You get a **table** (Δh, γₛ, volume, height, etc.).
          3. Pick a row to visualise and export the shape.
        - **Units**: length **m**, volume **m³**, γₛ **N/m**, ρ **kg/m³**, g **m/s²**.
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
    rho_m3 = st.number_input(
        "ρ - Density (kg/m³)",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

with col2:
    extra_slosh_height = st.number_input(
        "Extra collar height for sloshing (m)",
        min_value=0.0,
        value=0.10,
        step=0.01,
        help="Extra freeboard above Δh"
    )
    g_m3 = st.number_input(
        "g - Gravity (m/s²)",
        min_value=0.1,
        value=9.81,
        step=0.1
    )
    # No extra tube inputs for simple model

st.markdown("---")

st.header("🔍 Δh Range")

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
        "Δh step (m)",
        min_value=0.001,
        value=0.01,
        step=0.001
    )

st.markdown("---")

if st.button("🔬 Generate Solutions Table", type="primary", use_container_width=True):
    with st.spinner("Computing..."):
        try:
            # Genereer referentie druppel met willekeurige γₛ om h_cut te vinden
            ref_gamma = 35000.0
            df_ref = generate_droplet_shape(ref_gamma, rho_m3, g_m3, cut_percentage=0)
            h_cut_ref = find_height_for_diameter(df_ref, cut_diameter_m3)
            
            if h_cut_ref is None or np.isnan(h_cut_ref):
                st.error(f"❌ Cannot find height for diameter {cut_diameter_m3} m")
                st.stop()
            
            # Bepaal target kromming
            H_target = estimate_mean_curvature_at_height(df_ref, h_cut_ref)
            
            if H_target is None or np.isnan(H_target):
                st.error("❌ Cannot determine curvature")
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
                    
                    # Voeg vlakke top toe (EXACT zoals Methode 1)
                    target_radius = cut_diameter_m3 / 2.0
                    n_points = 30
                    # Plaats vlakke top aan de rechterkant [0, R] zodat deze naar rechts wijst
                    x_shifted_vals = np.linspace(0.0, target_radius, n_points)
                    top_points_data = []
                    x_max_current = df_cut['x-x_0'].max() if 'x-x_0' in df_cut.columns else 0.0
                    for x_sh in x_shifted_vals:
                        top_points_data.append({
                            'B': 1.0, 'C': 1.0, 'z': 0,
                            'x-x_0': x_sh + x_max_current,
                            'x_shifted': x_sh,
                            'h': h_cut_test
                        })
                    top_points = pd.DataFrame(top_points_data)
                    # Dedupliceer op ruwe x-kolom en hoogte zodat vlakke top een enkele lijn vormt
                    subset_cols = ['x-x_0', 'h'] if 'x-x_0' in df_cut.columns else ['x_shifted', 'h']
                    df_cut = pd.concat([df_cut, top_points], ignore_index=True).drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
                    
                    # Bereken metrics
                    metrics = get_droplet_metrics(df_cut)
                    
                    # NIEUWE LOGICA: volume_kraag moet exact gelijk zijn aan volume_afgekapt
                    # Bereken afgekapt volume (full - cut)
                    df_full_test = generate_droplet_shape(gamma_needed, rho_m3, g_m3, cut_percentage=0)
                    metrics_full = get_droplet_metrics(df_full_test)
                    volume_full = metrics_full.get('volume', 0.0)
                    volume_cut = metrics.get('volume', 0.0)
                    volume_afgekapt = volume_full - volume_cut
                    
                    # Los Δh op zodat volume_kraag(Δh) = volume_afgekapt
                    dh_result = find_delta_h_for_collar_volume(
                        target_volume=volume_afgekapt,
                        opening_diameter=cut_diameter_m3,
                        tube_diameter=float(tube_diameter_m3),
                        center_offset=float(tube_center_offset_m3),
                        tolerance=0.01,
                        max_iter=100
                    )
                    
                    # Update dh met de gevonden waarde
                    if dh_result['converged']:
                        dh_updated = dh_result['delta_h']
                    else:
                        dh_updated = dh  # Fallback naar originele dh
                    
                    # Bereken torus info met de geüpdatete dh
                    head_total = dh_updated + float(extra_slosh_height)
                    torus_info = compute_torus_from_head(
                        opening_diameter=cut_diameter_m3,
                        head_total=head_total,
                        wall_thickness=0.0,
                        safety_freeboard=float(extra_slosh_height)
                    )
                    # Voor 2D-visualisatie van de torus (kraag) opslaan zoals bij Methode 1
                    seam_h_eff = float(h_cut_test + dh_updated)  # ringniveau + dh = naad/seam
                    
                    solutions.append({
                        'Δh (m)': round(dh_updated, 4),  # Gebruik geüpdatete dh
                        'γₛ (N/m)': round(gamma_needed, 0),
                        'Volume (m³)': round(metrics.get('volume', 0), 2),
                        'Max height (m)': round(metrics.get('max_height', 0), 2),
                        'Base diameter (m)': round(metrics.get('bottom_diameter', 0), 2),
                        'Max diameter (m)': round(metrics.get('max_diameter', 0), 2),
                        'Torus water (m³)': round(torus_info.get('water_volume', 0), 2),
                        # Torus/kraag geometrie- en weergavevelden voor 2D
                        'torus_R_major': float(cut_diameter_m3) / 2.0,  # Exact: opening_diameter/2
                        'torus_r_top': head_total / 2.0,  # Simple half-torus: r = head_total/2
                        'torus_r_water': torus_info.get('r_water', 0.0),
                        'torus_head_total': torus_info.get('head_total', 0.0),
                        'delta_h_water': float(dh_updated),  # Gebruik geüpdatete dh
                        'h_seam_eff': seam_h_eff,
                    # Equivalent head volume over opening (A × Δh)
                    'equivalent_opening_volume': (np.pi * (cut_diameter_m3/2.0)**2) * float(dh_updated),
                    # Simple half‑torus displacement model: r = (Δh + sloshing)/2, displaced = 0.5 * 2π² R r²
                    'collar_displaced_volume': 0.5 * (2.0 * (np.pi ** 2) * (cut_diameter_m3/2.0) * (((dh_updated + float(extra_slosh_height)) / 2.0) ** 2)),
                        '_df': df_cut,
                        '_gamma': gamma_needed,
                        '_h_cut': h_cut_test,
                        'volume_afgekapt': volume_afgekapt,
                        'volume_kraag': dh_result['volume_achieved'],
                        'volume_match': dh_result['converged'],
                    })
                
                except Exception as e:
                    continue
            
            if not solutions:
                st.error("❌ No valid solutions found. Adjust parameters.")
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
    display_cols = ['Δh (m)', 'γₛ (N/m)', 'Volume (m³)', 'Max height (m)', 
                    'Base diameter (m)', 'Max diameter (m)', 'Torus water (m³)']
    
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
            format_func=lambda i: f"Δh = {st.session_state.solutions_table_m3[i]['Δh (m)']}m, γₛ = {st.session_state.solutions_table_m3[i]['γₛ (N/m)']:.0f} N/m",
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
        st.metric("Δh (m)", f"{st.session_state.selected_solution_m3['Δh (m)']:.4f}")
        st.metric("γₛ (N/m)", f"{st.session_state.selected_solution_m3['γₛ (N/m)']:.0f}")
        st.metric("Volume (m³)", f"{st.session_state.selected_solution_m3['Volume (m³)']:.2f}")
    
    with col2:
        st.metric("Max height (m)", f"{st.session_state.selected_solution_m3['Max height (m)']:.2f}")
        st.metric("Base diameter (m)", f"{st.session_state.selected_solution_m3['Base diameter (m)']:.2f}")
        st.metric("Equivalent head volume over opening (m³)", f"{st.session_state.selected_solution_m3.get('equivalent_opening_volume', 0):.2f}")
        # Net head water above opening: A*Δh - displaced collar volume
        try:
            net_head = float(st.session_state.selected_solution_m3.get('equivalent_opening_volume', 0)) - float(st.session_state.selected_solution_m3.get('collar_displaced_volume', 0))
        except Exception:
            net_head = 0.0
        st.metric("Net water above opening (m³)", f"{max(0.0, net_head):.2f}")
    
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
            gamma_val = st.session_state.selected_solution_m3['γₛ (N/m)']
            dh_val = st.session_state.selected_solution_m3['Δh (m)']
            st.download_button(
                "📥 Download STL",
                data=stl_data,
                file_name=f"droplet_dh{dh_val:.2f}_gamma{gamma_val:.0f}.stl",
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
                file_name=f"droplet_dh{dh_val:.2f}_gamma{gamma_val:.0f}.dxf",
                mime="application/dxf",
                use_container_width=True
            )

else:
    st.info("👆 Generate solutions and select one to visualise and export.")
