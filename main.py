"""
De Gennes Droplet Shape Calculator - Homepage
"""

import streamlit as st
from auth import require_password

st.set_page_config(
    page_title="De Gennes Droplet Shape Calculator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Simple password gate ----
def _require_password() -> None:
    """Stop rendering until the correct password is provided."""
    correct = "buitink"
    if st.session_state.get("auth_ok", False):
        return
    def _on_enter():
        if st.session_state.get("app_password", "") == correct:
            st.session_state["auth_ok"] = True
            try:
                del st.session_state["app_password"]
            except Exception:
                pass
        else:
            st.session_state["auth_ok"] = False
    st.sidebar.text_input("Password", type="password", key="app_password", on_change=_on_enter)
    if not st.session_state.get("auth_ok", False):
        st.warning("Enter password to continue.")
        st.stop()

require_password()

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