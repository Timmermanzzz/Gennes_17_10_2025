"""
Methode 1: Druppelvorm Berekenen
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
)
from visualisatie import create_2d_plot
from export import export_to_stl, export_to_dxf
import tempfile

st.set_page_config(
    page_title="Methode 1 - Druppelvorm Berekenen",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Methode 1 — Druppelvorm Berekenen")
st.markdown("Bereken druppelvormen met gegeven parameters en kies uw afkapopties.")

# Uitleg/Help
with st.expander("ℹ️ Uitleg — Hoe werkt Methode 1?", expanded=False):
    st.markdown(
        """
        - **Doel**: Bepaal de druppelvorm uit evenwicht van **Young–Laplace** (De Gennes) met opgegeven materiaal- en vloeistofparameters.
        - **Wat is γₛ?** Effectieve **membraanspanning** (N/m) in de huid. Geen E‑modulus en geen druk; het is de in‑vlak spanning per lengteeenheid die via **Δp = 2 γₛ H** de kromming bepaalt. Hogere γₛ ⇒ vlakker/strakker; lagere γₛ ⇒ ronder/boller.
        - **Invoer**:
          - **γₛ (N/m)**: membraanspanning/oppervlaktespanning
          - **ρ (kg/m³)**: dichtheid van de vloeistof
          - **g (m/s²)**: zwaartekracht
        - **Wat gebeurt er bij afkappen?**
          - Je snijdt de top open. Daardoor valt een deel van de **waterkolom** weg en verdwijnt een stuk **membraan**.
          - Gevolg: de **hydrostatische druk** bovenin is lager en de **kromming** aan de rand verandert. De druppel wordt slanker/lager dan het gesloten origineel.
        - **Hoe compenseren we dat? (kraag/torus)**
          - We plaatsen een **stijve ring**: die houdt de opening (diameter) exact vast — dus het membraan kan daar niet naar binnen/buiten schuiven.
          - We voegen een **kraag (donut/torus)** toe die we vullen met water tot een hoogte **Δh** boven de ring.
          - Dat water staat in verbinding met het reservoir en levert weer **ρ g Δh** extra druk op de rand.
          - Samen zorgt dit ervoor dat de **kromming aan de rand** weer overeenkomt met die van het **gesloten** reservoir op dezelfde hoogte.
          - Kort: de ring **fixeert de geometrie** (diameter), het water in de kraag **herstelt de drukkolom**.
        - **Afkapkeuzes**:
          - **Geen**: volledige (gesloten) druppel
          - **Afkap percentage**: snij top op een percentage van de hoogte
          - **Afkap diameter**: stel een vaste opening in (diameter). We zoeken de bijbehorende **afkaphoogte** en tekenen een **vlak** deksel op die hoogte.
        - **Eenheden**: lengte **m**, volume **m³**, γₛ **N/m**, ρ **kg/m³**, g **m/s²**.
        - **Uitvoer**: volume, maximale hoogte, basisdiameter, maximale diameter, (eventuele) afkapdiameter en kraagkenmerken.
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
        "γₛ - Oppervlaktespanning (N/m)",
        min_value=100.0,
        max_value=1000000.0,
        value=35000.0,
        step=1000.0,
        help="Oppervlaktespanningsparameter van het materiaal"
    )

with col2:
    rho = st.number_input(
        "ρ - Dichtheid (kg/m³)",
        min_value=1.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0,
        help="Dichtheid van de vloeistof"
    )

with col3:
    g = st.number_input(
        "g - Zwaartekracht (m/s²)",
        min_value=0.1,
        max_value=20.0,
        value=9.8,
        step=0.1,
        help="Gravitatieversnelling (standaard 9.8 op aarde)"
    )

# Row 2: Shape adjustment
col4, col5 = st.columns([1, 1])

with col4:
    st.subheader("Vorm aanpassing")
    
    cut_method = st.selectbox(
        "Afkap methode:",
        ["Geen afkap", "Afkap percentage", "Afkap diameter"],
        help="Kies hoe je de druppel wilt aanpassen"
    )
    
    use_diameter_mode = False
    use_percentage_mode = False
    
    if cut_method == "Afkap percentage":
        cut_percentage = st.slider(
            "Afkap percentage (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Percentage om van de bovenkant af te knippen"
        )
        cut_diameter = None
        use_percentage_mode = cut_percentage > 0
    elif cut_method == "Afkap diameter":
        cut_diameter = st.number_input(
            "Afkap diameter (m)",
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
    st.subheader("Constraints (optioneel)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_volume_constraint = st.toggle("Doelvolume afdwingen", value=False)
        target_volume = 0.0
        if use_volume_constraint:
            target_volume = st.number_input("Doel volume (m³)", min_value=0.0, value=1000.0, step=10.0)
    with col_c2:
        use_height_constraint = st.toggle("Doelhoogte afdwingen", value=False)
        target_height = 0.0
        if use_height_constraint:
            target_height = st.number_input("Doel hoogte (m)", min_value=0.0, value=3.3, step=0.01)
    
    st.markdown("")
    st.subheader("Kraag / torus (optioneel)")
    extra_slosh_height = st.number_input("Extra kraaghoogte voor klotsen (m)", min_value=0.0, value=0.10, step=0.01)

with col5:
    st.subheader("Actie")
    if st.button("🔬 Bereken Druppel", type="primary", use_container_width=True):
        with st.spinner("Berekening..."):
            try:
                # Generate full droplet first
                df_full = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                full_metrics = get_droplet_metrics(df_full)
                full_basis_diameter = full_metrics['bottom_diameter']
                full_max_diameter = full_metrics['max_diameter']
                
                df = df_full.copy()
                actual_cut_diameter = None
                
                if use_diameter_mode and cut_diameter > 0:
                    cut_at_height = find_height_for_diameter(df, cut_diameter)
                    if not np.isnan(cut_at_height):
                        df = df[df['h'] <= cut_at_height].copy()
                        target_radius = cut_diameter / 2.0
                        n_points = 30
                        x_shifted_vals = np.linspace(-target_radius, target_radius, n_points)
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
                
                st.session_state.df = df
                st.session_state.metrics = metrics
                st.session_state.physical_params = physical_params
                
                if use_diameter_mode and cut_diameter > 0:
                    st.success(f"✅ Reservoir met {cut_diameter:.1f}m opening berekend!")
                elif use_percentage_mode and cut_percentage > 0:
                    st.success(f"✅ Reservoir met {int(cut_percentage)}% afkap berekend!")
                elif use_volume_constraint and target_volume > 0 and not use_height_constraint:
                    st.success(f"✅ Doelvolume ≈ {target_volume:.1f} m³ gehaald!")
                elif use_height_constraint and target_height > 0:
                    st.success(f"✅ Doelhoogte ≈ {target_height:.2f} m gehaald!")
                else:
                    st.success("✅ Berekening succesvol!")
                
            except Exception as e:
                st.error(f"❌ Fout: {str(e)}")

st.markdown("---")

# Results
if st.session_state.df is not None:
    st.header("📊 Specificaties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Volume (m³)", f"{st.session_state.metrics.get('volume', 0):.2f}")
        st.metric("Maximale hoogte (m)", f"{st.session_state.metrics.get('max_height', 0):.2f}")
        st.metric("Maximale diameter (m)", f"{st.session_state.metrics.get('max_diameter', 0):.2f}")
    
    with col2:
        st.metric("Basis diameter (m)", f"{st.session_state.metrics.get('bottom_diameter', 0):.2f}")
        if cut_method != "Geen afkap":
            st.metric("Afkap diameter (m)", f"{st.session_state.metrics.get('top_diameter', 0):.2f}")
        else:
            st.metric("Afkap diameter (m)", "-")
        st.metric("γₛ (N/m)", f"{st.session_state.physical_params.get('gamma_s', 0):.0f}")
        
        if st.session_state.metrics.get('delta_h_water', 0) > 0:
            st.metric("Benodigde Δh (m)", f"{st.session_state.metrics.get('delta_h_water', 0):.2f}")
            if st.session_state.metrics.get('torus_head_total', 0) > 0:
                st.metric("Totale kraaghoogte (m)", f"{st.session_state.metrics.get('torus_head_total', 0):.2f}")
                st.metric("Torus water (m³)", f"{st.session_state.metrics.get('torus_water_volume', 0):.2f}")
    
    st.markdown("---")
    st.header("📈 Visualisatie")
    fig_2d = create_2d_plot(
        st.session_state.df,
        metrics=st.session_state.metrics,
        view="full",
        show_seam=False,
        show_cut_plane=True,
        cut_plane_h=st.session_state.metrics.get('h_cut', None)
    )
    st.plotly_chart(fig_2d, use_container_width=True)
    
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
            st.download_button("📥 Download STL", data=stl_data, file_name="druppel.stl", mime="application/octet-stream", use_container_width=True)
    
    with col_exp2:
        def gen_dxf():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                if export_to_dxf(st.session_state.df, tmp.name):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        dxf_data = gen_dxf()
        if dxf_data:
            st.download_button("📥 Download DXF", data=dxf_data, file_name="druppel.dxf", mime="application/dxf", use_container_width=True)

else:
    st.info("👆 Stel parameters in en klik 'Bereken Druppel' om te beginnen.")
