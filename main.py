"""
De Gennes Droplet Shape Calculator - Homepage
"""

import streamlit as st

st.set_page_config(
    page_title="De Gennes Droplet Shape Calculator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💧 De Gennes Droplet Shape Calculator")
st.markdown("*Based on Young–Laplace physics*")

st.markdown("---")

st.header("🎯 Welcome!")

st.markdown("""
This tool helps you compute and optimize droplet shapes using physical principles.

### **Two Methods Available:**

**📊 Method 1: Compute Droplet Shape**
- Compute shapes from given parameters (γₛ, ρ, g)
- Choose cut options: percentage, diameter, or no cut
- Add collar/torus to restore boundary conditions
- Export as STL or DXF

**🎯 Method 3: γₛ Optimization**
- Specify cut diameter
- App computes the optimal membrane tension (γₛ)
- Determine required collar head Δh
- Find the best combination for your design

---

### **How to Use**

Pick a method in the left sidebar and follow the instructions!
""")

st.markdown("---")

st.info("👈 **Select a method in the left menu to get started!**")

st.markdown("---")

st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "De Gennes Droplet Shape Calculator | Based on Young–Laplace physics"
    "</div>",
    unsafe_allow_html=True
)