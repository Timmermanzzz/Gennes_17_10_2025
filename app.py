"""
De Gennes Druppelvorm Calculator - Streamlit App
Simpele versie voor enkele druppel generatie.
"""

import streamlit as st
import pandas as pd
from solver import generate_droplet_shape, get_physical_parameters
from utils import (
    shift_x_coordinates,
    get_droplet_metrics,
    find_height_for_diameter,
    solve_gamma_for_volume,
    solve_gamma_for_height,
    compute_torus_from_head,
)
from visualisatie import create_2d_plot, create_3d_plot, create_metrics_table
from export import export_to_stl, export_to_dxf, get_export_filename
import os
import numpy as np

# Configuratie
st.set_page_config(
    page_title="De Gennes Druppelvorm Calculator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state first
if 'df' not in st.session_state:
    st.session_state.df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'physical_params' not in st.session_state:
    st.session_state.physical_params = None

# Enhanced title with calculation parameters display
st.title("💧 De Gennes Druppelvorm Calculator")
st.markdown("*Gebaseerd op Young-Laplace natuurkundige principes*")

# Show current calculation parameters if available
if st.session_state.df is not None and st.session_state.physical_params is not None:
    params = st.session_state.physical_params
    st.caption(f"γₛ = {params.get('gamma_s', 0):.0f} N/m | ρ = {params.get('rho', 0):.0f} kg/m³ | g = {params.get('g', 0):.1f} m/s² | κ = {params.get('kappa', 0):.4f} m⁻¹")

st.markdown("---")

# Parameters bovenaan - altijd custom
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

# Row 2: Shape adjustment and calculate button
col4, col5 = st.columns([1, 1])

with col4:
    st.subheader("Vorm aanpassing")
    
    # Keuzemenu voor afkap methode
    cut_method = st.selectbox(
        "Afkap methode:",
        ["Geen afkap", "Afkap percentage", "Afkap diameter"],
        help="Kies hoe je de druppel wilt aanpassen"
    )
    
    # Zorg dat de flags altijd gedefinieerd zijn
    use_diameter_mode = False
    use_percentage_mode = False

    if cut_method == "Afkap percentage":
        cut_percentage = st.slider(
            "Afkap percentage (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Percentage om van de bovenkant van de druppel af te knippen"
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
            help="Diameter van de opening (bovenkant) van het reservoir"
        )
        cut_percentage = 0
        use_diameter_mode = cut_diameter > 0
    else:  # Geen afkap
        cut_percentage = 0
        cut_diameter = 0
        use_diameter_mode = False
        use_percentage_mode = False

    st.markdown("")
    st.subheader("Constraints (optioneel)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_volume_constraint = st.toggle("Doelvolume afdwingen", value=False, help="Zoek γₛ zodat het volume gelijk wordt aan het doelvolume.")
        target_volume = 0.0
        if use_volume_constraint:
            target_volume = st.number_input("Doel volume (m³)", min_value=0.0, value=1000.0, step=10.0)
    with col_c2:
        use_height_constraint = st.toggle("Doelhoogte afdwingen", value=False, help="Zoek γₛ zodat de maximale hoogte gelijk wordt aan de doelhoogte.")
        target_height = 0.0
        if use_height_constraint:
            target_height = st.number_input("Doel hoogte (m)", min_value=0.0, value=3.3, step=0.01)

    # Kraag/torus instellingen
    st.subheader("Kraag / torus (optioneel)")
    
    # Info knop voor kraag methoden uitleg
    if st.button("ℹ️ Kraag methoden", help="Klik voor uitleg over de verschillende kraag methoden"):
        st.session_state.show_collar_explanation = not st.session_state.get('show_collar_explanation', False)
    
    # Toon uitleg als knop is ingedrukt
    if st.session_state.get('show_collar_explanation', False):
        with st.expander("🔬 Kraag Methoden - Fysica Uitleg", expanded=True):
            st.markdown("""
            **Het Probleem na Afkappen:**
            Wanneer je een druppel afkapt, verlies je:
            - **Hydrostatische druk** van de weggehaalde waterkolom  
            - **Membraan spanning** van het weggehaalde zeildoek
            
            Dit zorgt ervoor dat de druppel niet meer de optimale Young-Laplace kromming heeft.
            
            **Wat Gebeurt Er Praktisch met de Vorm?**
            Na afkappen zal de druppel **automatisch van vorm veranderen**:
            - **Slanker worden** - minder bolling aan de zijkanten
            - **Lager worden** - de totale hoogte neemt af
            - **Minder volume** - door het weggehaalde deel
            - **Verstoorde kromming** - niet meer de optimale druppelvorm
            
            **Waarom Dit Gebeurt:**
            - **Minder hydrostatische druk** → membraan trekt minder uit
            - **Minder watergewicht** → minder spanning op de wanden
            - **Young-Laplace evenwicht** wordt verstoord
            - **Resultaat:** druppel "krimpt" naar een nieuwe, minder optimale vorm
            
            **De Stijve Ring - Cruciaal Onderdeel:**
            Bij het afkappen wordt een **stijve ring** (bijv. stalen koord) door de rand gehaald:
            - **Opening diameter blijft vast** - kan niet rekken of krimpen
            - **Membraan kan niet samentrekken** op de rand
            - **Vorm verandering** gebeurt alleen in het vrije membraan
            - **Zonder ring** zou de opening ook krimpen en de hele vorm instorten
            
            **De Oplossing - Kraag Methoden:**
            
            **Methode 1: Torus/Donut Kraag**
            De torus herstelt de hydrostatische druk door:
            - Een **waterkolom** toe te voegen die dezelfde druk geeft als het weggehaalde deel
            - Het water in de torus staat **direct in verbinding** met het reservoir water
            - **Geen membraan** tussen torus en reservoir
            
            **Hoe het Werkt:**
            1. **Δh** = hoogte van weggehaalde waterkolom
            2. **Torus waterkolom** = exact dezelfde hydrostatische druk
            3. **Resultaat** = oorspronkelijke druppelkromming wordt hersteld
            
            **Praktisch:**
            - **Donut vorm** rond de opening (opgeblazen met lucht)
            - **Waterkanaal** in het midden voor de waterkolom
            - **Flexibel** - kan klotsen opvangen
            - **Efficiënt** - minimale waterhoeveelheid nodig
            - **Herstelt** de oorspronkelijke optimale vorm
            
            **Methode 2: [Wordt toegevoegd]**
            *Nieuwe methode in ontwikkeling...*
            """)
    
    extra_slosh_height = st.number_input("Extra kraaghoogte voor klotsen (m)", min_value=0.0, value=0.10, step=0.01, help="Extra vrije boord boven Δh.")
    # Wanddikte verwaarloosd volgens gebruiker (≈0). We houden r_water ≈ r_top.
    torus_wall_thickness = 0.0

with col5:
    st.subheader("Actie")
    if st.button("🔬 Bereken Druppel", type="primary", use_container_width=True):
        with st.spinner("Berekening..."):
            try:
                # Genereer volledige druppelvorm (zonder afkap eerst!)
                df_full = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                # x_shifted is al toegevoegd in generate_droplet_shape()
                
                # Bereken basis diameter van de VOLLEDIGE druppel (voor afkap)
                full_metrics = get_droplet_metrics(df_full)
                full_basis_diameter = full_metrics['bottom_diameter']
                full_max_diameter = full_metrics['max_diameter']
                
                # Start met de volledige druppel
                df = df_full.copy()
                
                # Als diameter mode: vind juiste afkaphoogte en kapt af
                actual_cut_diameter = None  # Sla de echte afkap diameter op
                if use_diameter_mode and cut_diameter > 0:
                    # Zoek hoogte waar de diameter gelijk is aan desired diameter
                    cut_at_height = find_height_for_diameter(df, cut_diameter)
                    
                    if np.isnan(cut_at_height):
                        st.warning(f"⚠️ Diameter {cut_diameter:.1f}m niet gevonden in druppel. Volledige druppel getoond.")
                    else:
                        # Filter druppel: houd alles ONDER/OP deze hoogte (h <= cut_at_height)
                        df = df[df['h'] <= cut_at_height].copy()
                        
                        # Voeg punten toe voor vlakke bovenkant (altijd toevoegen)
                        target_radius = cut_diameter / 2.0
                        n_points = 30
                        x_shifted_vals = np.linspace(-target_radius, target_radius, n_points)
                        top_points_data = []
                        # reconstructeer x-x_0 uit x_shifted + constante
                        x_max_current = df['x-x_0'].max() if 'x-x_0' in df.columns else 0.0
                        for x_sh in x_shifted_vals:
                            top_points_data.append({
                                'B': 1.0,
                                'C': 1.0,
                                'z': 0,
                                'x-x_0': x_sh + x_max_current,
                                'x_shifted': x_sh,
                                'h': cut_at_height
                            })
                        top_points = pd.DataFrame(top_points_data)
                        df = pd.concat([df, top_points], ignore_index=True).drop_duplicates(subset=['x_shifted', 'h'], keep='first').reset_index(drop=True)
                        
                        # Sla de echte afkap diameter op
                        actual_cut_diameter = cut_diameter

                # Als percentage mode: genereer direct met cut_percentage
                if use_percentage_mode and cut_percentage > 0:
                    df = generate_droplet_shape(gamma_s, rho, g, cut_percentage=int(cut_percentage))

                # Volume constraint: zoek gamma_s die target volume haalt met gekozen afkap
                if use_volume_constraint and target_volume > 0 and not use_height_constraint:
                    # Welke afkap instelling geldt
                    cut_pct = int(cut_percentage) if use_percentage_mode else 0
                    cut_diam = float(actual_cut_diameter or 0.0) if use_diameter_mode else 0.0
                    gamma_opt, df_opt, vol_opt = solve_gamma_for_volume(
                        target_volume=target_volume,
                        rho=rho,
                        g=g,
                        cut_percentage=cut_pct,
                        cut_diameter=cut_diam,
                    )
                    df = df_opt
                    gamma_s = gamma_opt

                # Hoogte constraint: zoek gamma_s die target hoogte haalt
                if use_height_constraint and target_height > 0:
                    cut_pct = int(cut_percentage) if use_percentage_mode else 0
                    cut_diam = float(actual_cut_diameter or 0.0) if use_diameter_mode else 0.0
                    gamma_opt, df_opt, h_opt = solve_gamma_for_height(
                        target_height=target_height,
                        rho=rho,
                        g=g,
                        cut_percentage=cut_pct,
                        cut_diameter=cut_diam,
                    )
                    df = df_opt
                    gamma_s = gamma_opt
                
                # Recompute full-shape metrics with the (possibly) updated gamma_s
                df_full_final = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                full_metrics_final = get_droplet_metrics(df_full_final)
                full_basis_diameter_final = full_metrics_final['bottom_diameter']
                full_max_diameter_final = full_metrics_final['max_diameter']
                full_max_height_final = full_metrics_final['max_height']

                # Bereken metrieken op de (eventueel afgekapte) vorm
                metrics = get_droplet_metrics(df)
                physical_params = get_physical_parameters(df, gamma_s, rho, g)
                
                # Overschrijf diameters met correcte waarden
                if actual_cut_diameter is not None:
                    metrics['top_diameter'] = actual_cut_diameter
                
                # Basis en maximale diameter altijd van de VOLLEDIGE vorm bij actuele γₛ
                metrics['bottom_diameter'] = full_basis_diameter_final
                metrics['max_diameter'] = full_max_diameter_final

                # Bereken kraagvulling Δh en seam-hoogte (alleen bij afkap)
                delta_h_water = 0.0
                seam_h = None
                if (use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0)) or (use_percentage_mode and cut_percentage > 0):
                    if use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0):
                        # herbereken afkaphoogte op basis van definitieve γₛ
                        cut_h_final = find_height_for_diameter(df_full_final, float(actual_cut_diameter))
                    else:
                        # percentage: huidige tophoogte na afkap is df['h'].max()
                        cut_h_final = float(df['h'].max()) if not df.empty else np.nan
                    if cut_h_final is not None and not np.isnan(cut_h_final):
                        delta_h_water = max(0.0, float(full_max_height_final) - float(cut_h_final))
                        seam_h = float(full_max_height_final)
                metrics['delta_h_water'] = delta_h_water
                metrics['h_seam_eff'] = seam_h if seam_h is not None else 0.0

                # Indien opening bekend, bereken torus geadviseerde geometrie/volume
                opening_diam_for_torus = None
                if use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0):
                    opening_diam_for_torus = float(actual_cut_diameter)
                elif use_percentage_mode and cut_percentage > 0:
                    # openingdiameter op cut_h_final
                    if 'x_shifted' in df_full_final.columns and cut_h_final is not None and not np.isnan(cut_h_final):
                        # diameter op cut_h_final in full shape
                        from utils import calculate_diameter_at_height
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
                
                # Opslaan in session state
                st.session_state.df = df
                st.session_state.metrics = metrics
                st.session_state.physical_params = physical_params
                
                # Feedback tonen
                if use_diameter_mode and cut_diameter > 0:
                    st.success(f"✅ Reservoir met {cut_diameter:.1f}m opening berekend!")
                elif use_percentage_mode and cut_percentage > 0:
                    st.success(f"✅ Reservoir met {int(cut_percentage)}% afkap berekend!")
                elif use_volume_constraint and target_volume > 0 and not use_height_constraint:
                    st.success(f"✅ Doelvolume ≈ {target_volume:.1f} m³ gehaald door γₛ aan te passen!")
                elif use_height_constraint and target_height > 0:
                    st.success(f"✅ Doelhoogte ≈ {target_height:.2f} m gehaald door γₛ aan te passen!")
                else:
                    st.success("✅ Berekening succesvol!")
                
            except Exception as e:
                st.error(f"❌ Fout bij berekening: {str(e)}")

st.markdown("---")

# Main content area - Nieuwe workflow layout
if st.session_state.df is not None:
    # 1. EERST: Alle specificaties
    st.header("📊 Specificaties")
    
    # Metrics in 2x3 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Volume (m³)",
            value=f"{st.session_state.metrics.get('volume', 0):.2f}"
        )
        st.metric(
            label="Maximale hoogte (m)",
            value=f"{st.session_state.metrics.get('max_height', 0):.2f}"
        )
        st.metric(
            label="Maximale diameter (m)",
            value=f"{st.session_state.metrics.get('max_diameter', 0):.2f}"
        )
    
    with col2:
        st.metric(
            label="Basis diameter (m)",
            value=f"{st.session_state.metrics.get('bottom_diameter', 0):.2f}"
        )
        
        # Toon Afkap diameter ALLEEN als werkelijk afgekapt
        if cut_method != "Geen afkap":
            st.metric(
                label="Afkap diameter (m)",
                value=f"{st.session_state.metrics.get('top_diameter', 0):.2f}"
            )
        else:
            st.metric(
                label="Afkap diameter (m)",
                value="-"
            )
        
        st.metric(
            label="Kappa (m⁻¹)",
            value=f"{st.session_state.physical_params.get('kappa', 0):.4f}"
        )
        st.metric(
            label="γₛ - Membraanspanning (N/m)",
            value=f"{st.session_state.physical_params.get('gamma_s', 0):.0f}"
        )
        # Toon geadviseerde kraagvulling (Δh)
        if st.session_state.metrics.get('delta_h_water', 0) > 0:
            st.metric(
                label="Benodigde kraag hoogte Δh (m)",
                value=f"{st.session_state.metrics.get('delta_h_water', 0):.2f}"
            )
            # Toon torus gegevens indien beschikbaar
            if st.session_state.metrics.get('torus_head_total', 0) > 0:
                st.metric(
                    label="Totale kraaghoogte (Δh + extra) (m)",
                    value=f"{st.session_state.metrics.get('torus_head_total', 0):.2f}"
                )
                st.metric(
                    label="Torus water (m³)",
                    value=f"{st.session_state.metrics.get('torus_water_volume', 0):.2f}"
                )
    
    st.markdown("---")
    
    # 2. Visualisaties (alleen 2D)
    st.header("📊 Visualisaties")
    st.subheader("2D Doorsnede")
    fig_2d = create_2d_plot(st.session_state.df, metrics=st.session_state.metrics)
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.markdown("---")
    
    # 3. ONDERAAN: Export mogelijkheden
    st.header("💾 Export Opties")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        # Generate STL file on-the-fly
        def generate_stl():
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                success = export_to_stl(st.session_state.df, tmp.name)
                if success:
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        stl_data = generate_stl()
        if stl_data:
            st.download_button(
                label="📥 Download STL",
                data=stl_data,
                file_name="druppel.stl",
                mime="application/octet-stream",
                help="Voor 3D-printen",
                use_container_width=True
            )
    
    with col_export2:
        # Generate DXF file on-the-fly
        def generate_dxf():
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                success = export_to_dxf(st.session_state.df, tmp.name)
                if success:
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        dxf_data = generate_dxf()
        if dxf_data:
            st.download_button(
                label="📥 Download DXF",
                data=dxf_data,
                file_name="druppel.dxf",
                mime="application/dxf",
                help="Voor CAD software",
                use_container_width=True
            )

else:
    # Welkomstscherm wanneer nog geen berekening is gedaan
    st.info("👆 Stel parameters hierboven in en klik op 'Bereken Druppel' om te beginnen.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "De Gennes Druppelvorm Calculator | Gebaseerd op Young-Laplace natuurkunde"
    "</div>",
    unsafe_allow_html=True
)