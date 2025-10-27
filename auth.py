import streamlit as st


def require_password(password: str = "buitink", prompt_label: str = "Password") -> bool:
    """Simple Streamlit password gate. Stops rendering until correct password.

    Returns True when authenticated.
    """
    if st.session_state.get("auth_ok", False):
        return True

    def _on_enter() -> None:
        if st.session_state.get("app_password", "") == password:
            st.session_state["auth_ok"] = True
            try:
                del st.session_state["app_password"]
            except Exception:
                pass
        else:
            st.session_state["auth_ok"] = False

    st.sidebar.text_input(prompt_label, type="password", key="app_password", on_change=_on_enter)
    if not st.session_state.get("auth_ok", False):
        st.warning("Enter password to continue.")
        st.stop()
    return True


