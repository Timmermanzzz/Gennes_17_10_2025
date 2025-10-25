"""
Method 1: Compute Droplet Shape
"""

import streamlit as st
import pandas as pd
import numpy as np
from solver import generate_droplet_shape, get_physical_parameters
from utils import (
    shift_x_coordinates,
    get_droplet_metrics,
    find_height_for_diameter,
    solve_gamma_for_volume,
    solve_gamma_for_height,
    compute_torus_from_head,
    calculate_diameter_at_height,
    compute_collar_segment_volume,
    find_delta_h_for_collar_volume,
)
from visualisatie import create_2d_plot, create_3d_plot
from export import export_to_stl, export_to_dxf
import tempfile

st.set_page_config(
    page_title="Method 1 - Compute Droplet Shape",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Method 1 — Compute Droplet Shape")
st.markdown("Compute droplet shapes for given parameters and choose your cut options.")

# Uitleg/Help
with st.expander("ℹ️ Help — How does Method 1 work?", expanded=False):
    st.markdown(
        """
        - **Goal**: Solve the **Young–Laplace** (De Gennes) equilibrium droplet for given material and fluid parameters.
        - **What is γₛ?** Effective **membrane tension** (N/m). Not E‑modulus, not pressure; the in‑plane tension that with **Δp = 2 γₛ H** sets curvature. Higher γₛ ⇒ flatter; lower γₛ ⇒ rounder.
        - **Inputs**:
          - **γₛ (N/m)**: membrane/surface tension
          - **ρ (kg/m³)**: fluid density
          - **g (m/s²)**: gravity
        - **When cutting the top**:
          - Opening removes part of the **water column** and some **membrane**.
          - Result: lower **hydrostatic head** near the top and different edge curvature; the droplet becomes slimmer/lower than the closed one.
        - **How we compensate (collar/torus)**
          - A **rigid ring** fixes the opening diameter so the membrane cannot move there.
          - A **collar (donut/torus)** is filled with water to **Δh** above the ring.
          - This restores **ρ g Δh** pressure at the edge to match the closed reference curvature.
        - **Cut options**:
          - **None**: full (closed) droplet
          - **Cut percentage**: slice off a percentage of the height
          - **Cut diameter**: set a fixed opening (diameter). We find the **cut height** and draw a **flat** lid at that height.
        - **Units**: length **m**, volume **m³**, γₛ **N/m**, ρ **kg/m³**, g **m/s²**.
        - **Outputs**: volume, max height, base diameter, max diameter, opening diameter, and collar properties.
        """
    )

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'physical_params' not in st.session_state:
    st.session_state.physical_params = None

st.markdown("---")

st.header("⚙️ Parameters")

# Row 1: Physical properties
col1, col2, col3 = st.columns(3)

with col1:
    gamma_s = st.number_input(
        "γₛ - Surface tension (N/m)",
        min_value=100.0,
        max_value=1000000.0,
        value=35000.0,
        step=1000.0,
        help="Oppervlaktespanningsparameter van het materiaal"
    )

with col2:
    rho = st.number_input(
        "ρ - Density (kg/m³)",
        min_value=1.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0,
        help="Dichtheid van de vloeistof"
    )

with col3:
    g = st.number_input(
        "g - Gravity (m/s²)",
        min_value=0.1,
        max_value=20.0,
        value=9.8,
        step=0.1,
        help="Gravitatieversnelling (standaard 9.8 op aarde)"
    )

# Row 2: Shape adjustment
col4, col5 = st.columns([1, 1])

with col4:
    st.subheader("Shape adjustment")
    
    cut_method = st.selectbox(
        "Cut method:",
        ["No cut", "Cut percentage", "Cut diameter"],
        help="Choose how you want to adjust the droplet"
    )
    
    use_diameter_mode = False
    use_percentage_mode = False
    
    if cut_method == "Cut percentage":
        cut_percentage = st.slider(
            "Cut percentage (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Percentage om van de bovenkant af te knippen"
        )
        cut_diameter = None
        use_percentage_mode = cut_percentage > 0
    elif cut_method == "Cut diameter":
        cut_diameter = st.number_input(
            "Cut diameter (m)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
            help="Diameter van de opening"
        )
        cut_percentage = 0
        use_diameter_mode = cut_diameter > 0
    else:
        cut_percentage = 0
        cut_diameter = 0
    
    st.markdown("")
    st.subheader("Constraints (optional)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_volume_constraint = st.toggle("Enforce target volume", value=False)
        target_volume = 0.0
        if use_volume_constraint:
            target_volume = st.number_input("Target volume (m³)", min_value=0.0, value=1000.0, step=10.0)
    with col_c2:
        use_height_constraint = st.toggle("Enforce target height", value=False)
        target_height = 0.0
        if use_height_constraint:
            target_height = st.number_input("Target height (m)", min_value=0.0, value=3.3, step=0.01)
    
    st.markdown("")
    st.subheader("Collar / torus (optional)")
    extra_slosh_height = st.number_input("Extra collar height for sloshing (m)", min_value=0.0, value=0.10, step=0.01)
    tube_diameter = st.number_input("Collar tube diameter (m)", min_value=0.0, value=0.50, step=0.01, help="Outer diameter of the collar tube")
    tube_center_offset = st.number_input("Collar center offset (m)", min_value=0.0, value=0.0, step=0.01, help="Offset of tube center from ring edge")

with col5:
    st.subheader("Action")
    if st.button("🔬 Compute Droplet", type="primary", use_container_width=True):
        with st.spinner("Computing..."):
            try:
                # Generate full droplet first
                df_full = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                full_metrics = get_droplet_metrics(df_full)
                full_basis_diameter = full_metrics['bottom_diameter']
                full_max_diameter = full_metrics['max_diameter']
                
                df = df_full.copy()
                actual_cut_diameter = None
                
                df_before_top = None  # Bewaar voor volume berekening
                if use_diameter_mode and cut_diameter > 0:
                    cut_at_height = find_height_for_diameter(df, cut_diameter)
                    if not np.isnan(cut_at_height):
                        df = df[df['h'] <= cut_at_height].copy()
                        # Bewaar profiel VOOR top-toevoeging voor volume berekening
                        df_before_top = df.copy()
                        target_radius = cut_diameter / 2.0
                        n_points = 30
                        # Plaats vlakke top aan de rechterkant [0, R] zodat deze naar rechts wijst
                        x_shifted_vals = np.linspace(0.0, target_radius, n_points)
                        top_points_data = []
                        x_max_current = df['x-x_0'].max() if 'x-x_0' in df.columns else 0.0
                        for x_sh in x_shifted_vals:
                            top_points_data.append({
                                'B': 1.0, 'C': 1.0, 'z': 0,
                                'x-x_0': x_sh + x_max_current,
                                'x_shifted': x_sh,
                                'h': cut_at_height
                            })
                        top_points = pd.DataFrame(top_points_data)
                        # Dedupliceer op ruwe x-kolom en hoogte zodat vlakke top een enkele lijn vormt
                        subset_cols = ['x-x_0', 'h'] if 'x-x_0' in df.columns else ['x_shifted', 'h']
                        df = pd.concat([df, top_points], ignore_index=True).drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
                        actual_cut_diameter = cut_diameter
                
                if use_percentage_mode and cut_percentage > 0:
                    df = generate_droplet_shape(gamma_s, rho, g, cut_percentage=int(cut_percentage))
                
                if use_volume_constraint and target_volume > 0 and not use_height_constraint:
                    cut_pct = int(cut_percentage) if use_percentage_mode else 0
                    cut_diam = float(actual_cut_diameter or 0.0) if use_diameter_mode else 0.0
                    gamma_opt, df_opt, vol_opt = solve_gamma_for_volume(
                        target_volume=target_volume, rho=rho, g=g,
                        cut_percentage=cut_pct, cut_diameter=cut_diam,
                    )
                    df = df_opt
                    gamma_s = gamma_opt
                
                if use_height_constraint and target_height > 0:
                    cut_pct = int(cut_percentage) if use_percentage_mode else 0
                    cut_diam = float(actual_cut_diameter or 0.0) if use_diameter_mode else 0.0
                    gamma_opt, df_opt, h_opt = solve_gamma_for_height(
                        target_height=target_height, rho=rho, g=g,
                        cut_percentage=cut_pct, cut_diameter=cut_diam,
                    )
                    df = df_opt
                    gamma_s = gamma_opt
                    # Voor enforce height: df_opt bevat al een correct afgekapte vorm, geen extra top nodig
                    if use_diameter_mode and cut_diam > 0:
                        actual_cut_diameter = cut_diam
                
                df_full_final = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                full_metrics_final = get_droplet_metrics(df_full_final)
                full_basis_diameter_final = full_metrics_final['bottom_diameter']
                full_max_diameter_final = full_metrics_final['max_diameter']
                full_max_height_final = full_metrics_final['max_height']
                
                metrics = get_droplet_metrics(df)
                physical_params = get_physical_parameters(df, gamma_s, rho, g)
                
                if actual_cut_diameter is not None:
                    metrics['top_diameter'] = actual_cut_diameter
                
                metrics['bottom_diameter'] = full_basis_diameter_final
                metrics['max_diameter'] = full_max_diameter_final
                
                delta_h_water = 0.0
                seam_h = None
                if (use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0)) or (use_percentage_mode and cut_percentage > 0):
                    if use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0):
                        cut_h_final = find_height_for_diameter(df_full_final, float(actual_cut_diameter))
                    else:
                        cut_h_final = float(df['h'].max()) if not df.empty else np.nan
                    if cut_h_final is not None and not np.isnan(cut_h_final):
                        delta_h_water = max(0.0, float(full_max_height_final) - float(cut_h_final))
                        seam_h = float(full_max_height_final)
                
                metrics['delta_h_water'] = delta_h_water
                metrics['h_seam_eff'] = seam_h if seam_h is not None else 0.0
                # Bewaar afkaphoogte voor duidelijke visual (cut-plane)
                if 'cut_h_final' in locals() and cut_h_final is not None and not np.isnan(cut_h_final):
                    metrics['h_cut'] = float(cut_h_final)
                else:
                    metrics['h_cut'] = 0.0
                
                opening_diam_for_torus = None
                if use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0):
                    opening_diam_for_torus = float(actual_cut_diameter)
                elif use_percentage_mode and cut_percentage > 0:
                    if 'x_shifted' in df_full_final.columns and cut_h_final is not None and not np.isnan(cut_h_final):
                        opening_diam_for_torus = float(calculate_diameter_at_height(df_full_final, cut_h_final))
                
                if opening_diam_for_torus is not None and delta_h_water > 0:
                    # NOUWEE LOGICA: volume_kraag moet exact gelijk zijn aan volume_afgekapt
                    volume_full = full_metrics_final.get('volume', 0.0)
                    # Gebruik df_before_top voor correcte volume berekening (zonder de toegevoegde top)
                    if df_before_top is not None:
                        metrics_before_top = get_droplet_metrics(df_before_top)
                        volume_cut = metrics_before_top.get('volume', 0.0)
                    else:
                        volume_cut = metrics.get('volume', 0.0)
                    volume_afgekapt = volume_full - volume_cut
                    
                    # Los Δh op zodat volume_kraag(Δh) = volume_afgekapt
                    dh_result = find_delta_h_for_collar_volume(
                        target_volume=volume_afgekapt,
                        opening_diameter=opening_diam_for_torus,
                        tube_diameter=float(tube_diameter),
                        center_offset=float(tube_center_offset),
                        tolerance=0.01,
                        max_iter=100
                    )
                    
                    # Update delta_h_water met de gevonden waarde
                    if dh_result['converged']:
                        delta_h_water = dh_result['delta_h']
                        metrics['delta_h_water'] = delta_h_water
                        head_total = float(delta_h_water) + float(extra_slosh_height)
                    else:
                        # Fallback naar oude logica als geen convergerende oplossing
                        head_total = float(delta_h_water) + float(extra_slosh_height)
                    
                    torus_info = compute_torus_from_head(opening_diameter=opening_diam_for_torus,
                                                         head_total=head_total,
                                                         wall_thickness=0.0,
                                                         safety_freeboard=float(extra_slosh_height))
                    metrics['torus_R_major'] = torus_info['R_major']
                    metrics['torus_r_top'] = torus_info['r_top']
                    metrics['torus_r_water'] = torus_info['r_water']
                    metrics['torus_head_total'] = torus_info['head_total']
                    metrics['torus_water_volume'] = torus_info['water_volume']
                    
                    # Simple half‑torus displacement model (per user spec):
                    # r = (Δh + sloshing)/2, R_major = opening_diameter/2, displaced = portion inside head region
                    import math
                    R_major = float(opening_diam_for_torus) / 2.0
                    head_total_clean = float(delta_h_water) + float(extra_slosh_height)
                    r_simple = head_total_clean / 2.0
                    # Full torus volume
                    v_torus_simple = 2.0 * (math.pi ** 2) * R_major * (r_simple ** 2)
                    # Only the portion that sits in the head region (half‑torus) is displaced
                    displaced_simple = 0.5 * v_torus_simple
                    # Equivalent head volume A × Δh
                    opening_area = float(np.pi) * (float(opening_diam_for_torus) / 2.0) ** 2
                    eq_vol = opening_area * float(delta_h_water)
                    metrics['equivalent_opening_volume'] = eq_vol
                    metrics['head_volume_net'] = max(0.0, eq_vol - displaced_simple)
                    metrics['volume_afgekapt'] = volume_afgekapt
                    metrics['volume_kraag_match'] = dh_result['converged']
                
                st.session_state.df = df
                st.session_state.metrics = metrics
                st.session_state.physical_params = physical_params
                
                if use_diameter_mode and cut_diameter > 0:
                    st.success(f"✅ Reservoir with {cut_diameter:.1f} m opening computed!")
                elif use_percentage_mode and cut_percentage > 0:
                    st.success(f"✅ Reservoir with {int(cut_percentage)}% cut computed!")
                elif use_volume_constraint and target_volume > 0 and not use_height_constraint:
                    st.success(f"✅ Target volume ≈ {target_volume:.1f} m³ achieved!")
                elif use_height_constraint and target_height > 0:
                    st.success(f"✅ Target height ≈ {target_height:.2f} m achieved!")
                else:
                    st.success("✅ Computation successful!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")

# Results
if st.session_state.df is not None:
    st.header("📊 Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Volume (m³)", f"{st.session_state.metrics.get('volume', 0):.2f}")
        st.metric("Max height (m)", f"{st.session_state.metrics.get('max_height', 0):.2f}")
        st.metric("Max diameter (m)", f"{st.session_state.metrics.get('max_diameter', 0):.2f}")
    
    with col2:
        st.metric("Base diameter (m)", f"{st.session_state.metrics.get('bottom_diameter', 0):.2f}")
        if cut_method != "No cut":
            st.metric("Opening diameter (m)", f"{st.session_state.metrics.get('top_diameter', 0):.2f}")
        else:
            st.metric("Opening diameter (m)", "-")
        st.metric("γₛ (N/m)", f"{st.session_state.physical_params.get('gamma_s', 0):.0f}")
        
        if st.session_state.metrics.get('delta_h_water', 0) > 0:
            st.metric("Required Δh (m)", f"{st.session_state.metrics.get('delta_h_water', 0):.2f}")
            if st.session_state.metrics.get('torus_head_total', 0) > 0:
                st.metric("Total collar head (m)", f"{st.session_state.metrics.get('torus_head_total', 0):.2f}")
                # Toon kraagvolume (moet gelijk zijn aan afgekapt volume)
                volume_afgekapt = st.session_state.metrics.get('volume_afgekapt', 0)
                if volume_afgekapt > 0:
                    st.metric("Collar volume (m³)", f"{volume_afgekapt:.2f}")
    
    st.markdown("---")
    st.header("📈 Visualisation")
    st.subheader("2D Cross-section")
    fig_2d = create_2d_plot(
        st.session_state.df,
        metrics=st.session_state.metrics,
        view="full",
        show_seam=False,
        show_cut_plane=True,
        cut_plane_h=st.session_state.metrics.get('h_cut', None)
    )
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.subheader("3D Model")
    fig_3d = create_3d_plot(
        st.session_state.df,
        metrics=st.session_state.metrics,
        title="Droplet 3D Model"
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    st.header("💾 Export")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        def gen_stl():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                if export_to_stl(st.session_state.df, tmp.name):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        stl_data = gen_stl()
        if stl_data:
            st.download_button("📥 Download STL", data=stl_data, file_name="droplet.stl", mime="application/octet-stream", use_container_width=True)
    
    with col_exp2:
        def gen_dxf():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                if export_to_dxf(st.session_state.df, tmp.name):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        dxf_data = gen_dxf()
        if dxf_data:
            st.download_button("📥 Download DXF", data=dxf_data, file_name="droplet.dxf", mime="application/dxf", use_container_width=True)

else:
    st.info("👆 Set parameters and click 'Compute Droplet' to begin.")
