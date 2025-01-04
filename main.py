import streamlit as st
from analysis import main
from chatbot import ai_chatbot_page

st.set_page_config(page_title="Group Comparison Analysis", layout="wide", page_icon="📈")

# Initialize session state for stock analysis completion
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

# Navigation between pages
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Group Analysis", "AI Chatbot"],
    index=0 if not st.session_state["analysis_complete"] else 1,  # Default to stock analysis if incomplete
)

# Render the selected page
if page == "Group Analysis":
    main()
elif page == "AI Chatbot":
    if st.session_state["analysis_complete"] ==True:
        ai_chatbot_page()
    else:
        st.error("You must complete the Portfolio Analysis before accessing the AI Chatbot.")