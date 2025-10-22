"""
De Gennes Druppelvorm Calculator - Homepage
"""

import streamlit as st

st.set_page_config(
    page_title="De Gennes Druppelvorm Calculator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💧 De Gennes Druppelvorm Calculator")
st.markdown("*Gebaseerd op Young-Laplace natuurkundige principes*")

st.markdown("---")

st.header("🎯 Welkom!")

st.markdown("""
Dit gereedschap helpt je om druppelvormen te berekenen en optimaliseren op basis van fysische principes.

### **Twee Methoden Beschikbaar:**

**📊 Methode 1: Druppelvorm Berekenen**
- Bereken druppelvormen met gegeven parameters (γₛ, ρ, g)
- Kies afkap-opties: percentage, diameter, of geen afkap
- Voeg kraag/torus toe voor optimale herstelling
- Exporteer als STL of DXF

**🎯 Methode 3: γₛ Optimalisatie**
- Geef afkapdiameter op
- App berekent optimale membraanspanning (γₛ)
- Bepaal benodigde kraagvulling Δh
- Vind de beste combinatie voor je ontwerp

---

### **Hoe Gebruikt**

Kies een methode in het menu links (sidebar) en volg de instructies!
""")

st.markdown("---")

st.info("👈 **Selecteer een methode in het menu links om te beginnen!**")

st.markdown("---")

st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "De Gennes Druppelvorm Calculator | Gebaseerd op Young-Laplace natuurkunde"
    "</div>",
    unsafe_allow_html=True
)