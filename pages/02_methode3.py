"""
Method 3: Volume-Based Collar Design
"""

import streamlit as st
from auth import require_password
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import generate_droplet_shape, get_physical_parameters
from utils import (
    get_droplet_metrics,
    find_height_for_diameter,
    calculate_diameter_at_height,
    solve_gamma_for_cut_volume_match
)
from visualisatie import create_2d_plot, create_3d_plot
from export import export_to_stl, export_to_dxf, export_to_step, export_to_3dm
from pdf_export import export_to_pdf
import tempfile
from io import BytesIO

st.set_page_config(
    page_title="Method 3 - γₛ Optimization",
    page_icon="🎯",
    layout="wide"
)

require_password()

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
    st.caption("γₛ wordt per oplossing berekend; geen invoer nodig.")

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
            # Genereer druppel met een start γₛ (wordt verderop per oplossing herberekend)
            df_full = generate_droplet_shape(35000.0, rho_m3, g_m3, cut_percentage=0)
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
            x_shifted_vals = np.linspace(-target_radius, 0.0, n_points)
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
                # Fysisch: B = A_open*h - 2πR*A_segment(r, h)
                water_height = max(0.0, tube_diam - float(sloshing_m3))
                A_open = np.pi * (R_major ** 2)
                # segment area
                if water_height <= 0.0:
                    A_seg = 0.0
                elif water_height >= 2.0 * r_tube:
                    A_seg = np.pi * (r_tube ** 2)
                else:
                    A_seg = (
                        (r_tube ** 2) * np.arccos((r_tube - water_height) / r_tube)
                        - (r_tube - water_height) * np.sqrt(max(0.0, 2.0 * r_tube * water_height - water_height ** 2))
                    )
                V_disp = 2.0 * np.pi * R_major * A_seg
                collar_vol_phys = max(0.0, A_open * water_height - V_disp)
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
                # Recompute cut height for this gamma and add a flat top (0..R) for consistent visuals
                try:
                    h_cut_gamma = float(df_cut_gamma['h'].max()) if df_cut_gamma is not None and not df_cut_gamma.empty else float('nan')
                except Exception:
                    h_cut_gamma = float('nan')
                df_vis = df_cut_gamma
                try:
                    if df_cut_gamma is not None and not df_cut_gamma.empty and not np.isnan(h_cut_gamma):
                        n_points = 30
                        x_shifted_vals = np.linspace(-R_major, 0.0, n_points)
                        x_max_current = df_cut_gamma['x-x_0'].max() if 'x-x_0' in df_cut_gamma.columns else 0.0
                        top_points_gamma = pd.DataFrame({
                            'B': 1.0, 'C': 1.0, 'z': 0,
                            'x-x_0': x_shifted_vals + x_max_current,
                            'x_shifted': x_shifted_vals,
                            'h': h_cut_gamma
                        })
                        subset_cols_gamma = ['x-x_0', 'h'] if 'x-x_0' in df_cut_gamma.columns else ['x_shifted', 'h']
                        df_vis = pd.concat([df_cut_gamma, top_points_gamma], ignore_index=True).drop_duplicates(subset=subset_cols_gamma, keep='first').reset_index(drop=True)
                except Exception:
                    df_vis = df_cut_gamma

                metrics = get_droplet_metrics(df_vis)
                droplet_h = float(metrics.get('max_height', 0) or 0.0)
                total_h = droplet_h + float(tube_diam)
                entry = {
                    'Tube diameter (m)': round(float(tube_diam), 3),
                    'Collar volume (m³)': round(float(collar_vol_phys), 2),
                    'Cut volume (m³)': round(float(cut_vol_gamma), 2),
                    'Volume match (%)': round(100.0 * (cut_vol_gamma / max(collar_vol_phys, 1e-9)), 1),
                    'γₛ match (N/m)': round(float(gamma_match), 1),
                    'Reservoir volume (m³)': round(metrics.get('volume', 0), 2),
                    'Total volume (m³)': round(metrics.get('volume', 0) + float(collar_vol_phys), 2),
                    'Droplet height (m)': round(droplet_h, 2),
                    'Total height (m)': round(total_h, 2),
                    'Base diameter (m)': round(metrics.get('bottom_diameter', 0), 2),
                    'Max diameter (m)': round(metrics.get('max_diameter', 0), 2),
                    # Intern voor visualisatie/export van de selectie
                    'torus_R_major': R_major,
                    'torus_r_top': r_tube,
                    'torus_r_water': r_tube,
                    'torus_water_volume': collar_vol_phys,
                    'h_seam_eff': float(h_cut_gamma if not np.isnan(h_cut_gamma) else h_cut),
                    'collar_tube_diameter': float(tube_diam),
                    '_df': df_vis,
                    '_gamma': float(gamma_match),
                    '_h_cut': float(h_cut_gamma if not np.isnan(h_cut_gamma) else h_cut),
                    'volume_afgekapt': float(cut_vol_gamma),
                    'volume_kraag': float(collar_vol_phys),
                }
                # Waterline at seam + (tube height - sloshing)
                entry['h_waterline'] = float((h_cut_gamma if not np.isnan(h_cut_gamma) else h_cut) + water_height)
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
                    'γₛ match (N/m)', 'Reservoir volume (m³)', 'Total volume (m³)', 'Droplet height (m)', 'Total height (m)', 'Base diameter (m)', 'Max diameter (m)']
    
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
    
    # Volumes (like Method 1)
    st.subheader("💧 Volumes")
    vcol1, vcol2, vcol3 = st.columns(3)
    droplet_vol_sel = float(st.session_state.selected_solution_m3['Reservoir volume (m³)'])
    collar_vol_sel = float(st.session_state.selected_solution_m3['Collar volume (m³)'])
    total_vol_sel = droplet_vol_sel + collar_vol_sel
    with vcol1:
        st.metric("Droplet Volume (m³)", f"{droplet_vol_sel:.2f}")
    with vcol2:
        st.metric("Collar Volume (m³)", f"{collar_vol_sel:.2f}")
    with vcol3:
        st.metric("Total Volume (m³)", f"{total_vol_sel:.2f}")
    vcol4, vcol5 = st.columns(2)
    with vcol4:
        st.metric("Cut volume (m³)", f"{st.session_state.selected_solution_m3['Cut volume (m³)']:.2f}")
    with vcol5:
        st.metric("Volume match (%)", f"{st.session_state.selected_solution_m3['Volume match (%)']:.1f}")

    # Geometry section
    st.subheader("📐 Geometry")
    g1, g2, g3 = st.columns(3)
    opening_diam_sel = 2.0 * float(st.session_state.selected_solution_m3.get('torus_R_major', 0.0) or 0.0)
    with g1:
        st.metric("Opening diameter (m)", f"{opening_diam_sel:.2f}")
        st.metric("Tube diameter (m)", f"{st.session_state.selected_solution_m3['Tube diameter (m)']:.3f}")
    with g2:
        st.metric("Droplet height (m)", f"{st.session_state.selected_solution_m3['Droplet height (m)']:.2f}")
        st.metric("Total height (m)", f"{st.session_state.selected_solution_m3['Total height (m)']:.2f}")
    with g3:
        st.metric("Base diameter (m)", f"{st.session_state.selected_solution_m3['Base diameter (m)']:.2f}")
        st.metric("Max diameter (m)", f"{st.session_state.selected_solution_m3['Max diameter (m)']:.2f}")

    # Material section
    st.subheader("⚙️ Material")
    st.metric("γₛ match (N/m)", f"{st.session_state.selected_solution_m3['γₛ match (N/m)']:.1f}")
    
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
    
    col_exp1, col_exp2, col_exp3, col_exp4, col_exp5 = st.columns(5)
    
    with col_exp1:
        def gen_stl():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                if export_to_stl(st.session_state.df_selected_m3, tmp.name, metrics=st.session_state.metrics_selected_m3):
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
                if export_to_dxf(st.session_state.df_selected_m3, tmp.name, metrics=st.session_state.metrics_selected_m3):
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

    with col_exp3:
        if st.button("📄 Generate PDF (A3)", use_container_width=True):
            pdf_bytes = None
            err = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    # In method 3 hebben we meestal geen los physical_params; geef None mee
                    export_to_pdf(st.session_state.df_selected_m3, st.session_state.metrics_selected_m3, tmp.name, physical_params=None)
                    with open(tmp.name, 'rb') as f:
                        pdf_bytes = f.read()
            except Exception as e:
                err = str(e)
            if pdf_bytes:
                tube_val = st.session_state.selected_solution_m3['Tube diameter (m)']
                st.download_button(
                    "📥 Download PDF (A3)",
                    data=pdf_bytes,
                    file_name=f"droplet_tube{tube_val:.2f}m.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error(f"PDF genereren mislukt. Controleer of reportlab is geïnstalleerd. Fout: {err}")

    with col_exp4:
        def gen_step():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as tmp:
                ok = export_to_step(st.session_state.df_selected_m3, tmp.name, metrics=st.session_state.metrics_selected_m3)
                if ok:
                    with open(tmp.name, 'rb') as f:
                        return f.read()
                return None
        if st.button("↗️ Download STEP", use_container_width=True):
            step_data = gen_step()
            if step_data:
                tube_val = st.session_state.selected_solution_m3['Tube diameter (m)']
                st.download_button(
                    "📥 STEP",
                    data=step_data,
                    file_name=f"droplet_tube{tube_val:.2f}m.step",
                    mime="application/step",
                    use_container_width=True
                )
            else:
                st.error("STEP export mislukt. Controleer of pythonocc-core is geïnstalleerd.")

    with col_exp5:
        def gen_3dm():
            import os, tempfile
            fd, tmp_path = tempfile.mkstemp(suffix='.3dm')
            os.close(fd)
            ok, err = export_to_3dm(st.session_state.df_selected_m3, tmp_path, metrics=st.session_state.metrics_selected_m3)
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
                tube_val = st.session_state.selected_solution_m3['Tube diameter (m)']
                st.download_button(
                    "📥 3DM",
                    data=data3dm,
                    file_name=f"droplet_tube{tube_val:.2f}m.3dm",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            else:
                st.error(f"3DM export mislukt. {err or 'Controleer of rhino3dm is geïnstalleerd.'}")

else:
    st.info("👆 Generate solutions and select one to visualise and export.")
