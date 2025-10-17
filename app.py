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

# Titel
st.title("💧 De Gennes Druppelvorm Calculator")
st.markdown("*Gebaseerd op Young-Laplace natuurkundige principes*")
st.markdown("---")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'physical_params' not in st.session_state:
    st.session_state.physical_params = None

# Parameters bovenaan de pagina
st.header("⚙️ Parameters")

# Maak kolommen voor parameters
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Fysische eigenschappen")
    gamma_s = st.number_input(
        "γₛ - Oppervlaktespanning (N/m)",
        min_value=100.0,
        max_value=1000000.0,
        value=35000.0,
        step=1000.0,
        help="Oppervlaktespanningsparameter van het materiaal"
    )
    
    rho = st.number_input(
        "ρ - Dichtheid (kg/m³)",
        min_value=1.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0,
        help="Dichtheid van de vloeistof"
    )
    
    g = st.number_input(
        "g - Zwaartekracht (m/s²)",
        min_value=0.1,
        max_value=20.0,
        value=9.8,
        step=0.1,
        help="Gravitatieversnelling (standaard 9.8 op aarde)"
    )

with col2:
    st.subheader("Vorm aanpassing")
    cut_percentage = st.slider(
        "Afkap percentage (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
        help="Percentage om van de bovenkant van de druppel af te knippen"
    )

with col3:
    st.subheader("Actie")
    # Bereken knop
    if st.button("🔬 Bereken Druppel", type="primary", use_container_width=True):
        with st.spinner("Druppelvorm wordt berekend..."):
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

# Main content area - Single page layout
if st.session_state.df is not None:
    # Belangrijkste specs direct onder parameters
    st.header("📊 Belangrijkste Specificaties")
    
    # Toon alleen de belangrijkste metrieken in een compacte layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Volume",
            value=f"{st.session_state.metrics.get('volume', 0):.2f} m³"
        )
    
    with col2:
        st.metric(
            label="Maximale hoogte",
            value=f"{st.session_state.metrics.get('max_height', 0):.2f} m"
        )
    
    with col3:
        st.metric(
            label="Maximale diameter",
            value=f"{st.session_state.metrics.get('max_diameter', 0):.2f} m"
        )
    
    with col4:
        st.metric(
            label="Basis diameter",
            value=f"{st.session_state.metrics.get('bottom_diameter', 0):.2f} m"
        )
    
    with col5:
        st.metric(
            label="Top diameter",
            value=f"{st.session_state.metrics.get('top_diameter', 0):.2f} m"
        )
    
    st.markdown("---")
    
    # 2D Visualisatie sectie
    st.header("📊 2D Visualisatie")
    st.subheader("Doorsnede van Druppelvorm")
    
    # Maak 2D plot
    fig_2d = create_2d_plot(st.session_state.df)
    st.plotly_chart(fig_2d, use_container_width=True)
    
    # Optioneel: 3D visualisatie
    with st.expander("🎲 Toon 3D Model"):
        fig_3d = create_3d_plot(st.session_state.df)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    
    # Volledige specificaties sectie
    st.header("📋 Volledige Specificaties")
    
    # Toon complete metrics tabel
    metrics_html = create_metrics_table(
        st.session_state.metrics,
        st.session_state.physical_params
    )
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Export sectie
    st.header("💾 Export Opties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### STL Export")
        st.markdown("Voor 3D-printen en mesh-bewerking")
        
        if st.button("📥 Download als STL", use_container_width=True):
            filepath = get_export_filename("druppel", ".stl", "exports")
            success = export_to_stl(st.session_state.df, filepath)
            
            if success and os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    st.download_button(
                        label="⬇️ Download STL bestand",
                        data=f.read(),
                        file_name=os.path.basename(filepath),
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                st.success("✅ STL bestand gegenereerd!")
            else:
                st.error("❌ Fout bij STL export")
    
    with col2:
        st.markdown("#### DXF Export")
        st.markdown("Voor CAD software (AutoCAD, etc.)")
        
        if st.button("📥 Download als DXF", use_container_width=True):
            filepath = get_export_filename("druppel", ".dxf", "exports")
            success = export_to_dxf(st.session_state.df, filepath)
            
            if success and os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    st.download_button(
                        label="⬇️ Download DXF bestand",
                        data=f.read(),
                        file_name=os.path.basename(filepath),
                        mime="application/dxf",
                        use_container_width=True
                    )
                st.success("✅ DXF bestand gegenereerd!")
            else:
                st.error("❌ Fout bij DXF export")

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

