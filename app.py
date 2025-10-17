"""
De Gennes Druppelvorm Calculator - Streamlit App
Simpele versie voor enkele druppel generatie.
"""

import streamlit as st
import pandas as pd
from solver import generate_droplet_shape, get_physical_parameters
from utils import shift_x_coordinates, get_droplet_metrics
from visualisatie import create_2d_plot, create_3d_plot, create_metrics_table
from export import export_to_stl, export_to_dxf, get_export_filename
import os

# Configuratie
st.set_page_config(
    page_title="De Gennes Druppelvorm Calculator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced title with calculation parameters display
st.title("💧 De Gennes Druppelvorm Calculator")
st.markdown("*Gebaseerd op Young-Laplace natuurkundige principes*")

# Show current calculation parameters if available
if st.session_state.df is not None and st.session_state.physical_params is not None:
    params = st.session_state.physical_params
    st.caption(f"γₛ = {params.get('gamma_s', 0):.0f} N/m | ρ = {params.get('rho', 0):.0f} kg/m³ | g = {params.get('g', 0):.1f} m/s² | κ = {params.get('kappa', 0):.4f} m⁻¹")

st.markdown("---")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'physical_params' not in st.session_state:
    st.session_state.physical_params = None

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
    cut_percentage = st.slider(
        "Afkap percentage (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
        help="Percentage om van de bovenkant van de druppel af te knippen"
    )

with col5:
    if st.button("🔬 Bereken Druppel", type="primary", use_container_width=True):
        with st.spinner("Berekening..."):
            try:
                # Genereer druppelvorm
                df = generate_droplet_shape(gamma_s, rho, g, cut_percentage)
                
                # Voeg x_shifted toe
                df = shift_x_coordinates(df)
                
                # Bereken metrieken
                metrics = get_droplet_metrics(df)
                physical_params = get_physical_parameters(df, gamma_s, rho, g)
                
                # Opslaan in session state
                st.session_state.df = df
                st.session_state.metrics = metrics
                st.session_state.physical_params = physical_params
                
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
            label="Volume",
            value=f"{st.session_state.metrics.get('volume', 0):.2f}",
            help="m³"
        )
        st.metric(
            label="Maximale hoogte",
            value=f"{st.session_state.metrics.get('max_height', 0):.2f}",
            help="m"
        )
        st.metric(
            label="Maximale diameter",
            value=f"{st.session_state.metrics.get('max_diameter', 0):.2f}",
            help="m"
        )
    
    with col2:
        st.metric(
            label="Basis diameter",
            value=f"{st.session_state.metrics.get('bottom_diameter', 0):.2f}",
            help="m"
        )
        st.metric(
            label="Top diameter",
            value=f"{st.session_state.metrics.get('top_diameter', 0):.2f}",
            help="m"
        )
        st.metric(
            label="Kappa (κ)",
            value=f"{st.session_state.physical_params.get('kappa', 0):.4f}",
            help="m⁻¹"
        )
    
    st.markdown("---")
    
    # 2. DAARNA: Visualisaties (2D links, 3D rechts)
    st.header("📊 Visualisaties")
    
    col_2d, col_3d = st.columns(2)
    
    with col_2d:
        st.subheader("2D Doorsnede")
        fig_2d = create_2d_plot(st.session_state.df)
        st.plotly_chart(fig_2d, use_container_width=True)
    
    with col_3d:
        st.subheader("3D Model")
        fig_3d = create_3d_plot(st.session_state.df)
        st.plotly_chart(fig_3d, use_container_width=True)
    
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
    
    # Voorbeeldafbeelding of instructies
    st.markdown("""
    ## Welkom bij de De Gennes Druppelvorm Calculator
    
    Deze applicatie helpt je bij het ontwerpen en visualiseren van druppelvormen 
    op basis van natuurkundige principes.
    
    ### Snelstart:
    1. **Stel fysische parameters in** hierboven (of gebruik de standaardwaarden)
    2. **Pas eventueel het afkap percentage aan** voor een vlakke bovenkant
    3. **Klik op 'Bereken Druppel'** om de vorm te genereren
    4. **Bekijk de resultaten** direct op deze pagina
    5. **Download STL/DXF** bestanden voor verder gebruik
    
    ### Standaardwaarden:
    - γₛ = 35000 N/m (typisch voor water)
    - ρ = 1000 kg/m³ (water)
    - g = 9.8 m/s² (zwaartekracht op aarde)
    - Afkap = 0% (volledige druppel)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "De Gennes Druppelvorm Calculator | Gebaseerd op Young-Laplace natuurkunde"
    "</div>",
    unsafe_allow_html=True
)

