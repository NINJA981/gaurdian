import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="FORGE-Guard | Manual",
    page_icon="📖",
    layout="wide"
)

def load_css():
    css_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def main():
    st.markdown("""
    <div class="forge-header">
        <h1>📖 Instruction Manual</h1>
        <p>FORGE-Guard Setup and Usage Guide</p>
    </div>
    """, unsafe_allow_html=True)
    
    manual_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "manual.md")
    
    if os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as f:
            content = f.read()
            st.markdown(content)
    else:
        st.error("Manual file not found at " + manual_path)
    
    if st.sidebar.button("🔙 Back to Dashboard"):
        st.info("To return, simply navigate to the main dashboard URL.")

if __name__ == "__main__":
    main()
