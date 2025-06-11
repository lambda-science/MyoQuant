import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(
    page_title="MyoQuant",
    page_icon="🔬",
)

navigation_dict = {
    "MyoQuant": [
        st.Page(
            page="pages/home.py",
            title="Home",
            icon="🏠",
            default=True,
        ),
        st.Page(
            page="pages/ATP_staining.py",
            title="ATP Staining",
            icon="🟣",
            default=False,
        ),
        st.Page(
            page="pages/HE_staining.py",
            title="HE Staining",
            icon="🔴",
            default=False,
        ),
        st.Page(
            page="pages/SDH_staining.py",
            title="SDH Staining",
            icon="🔵",
            default=False,
        ),
        st.Page(
            page="pages/Fluo_staining.py",
            title="Fluo Staining",
            icon="🟢",
            default=False,
        ),
    ],
}
selected_page = st.navigation(navigation_dict, position="sidebar")
selected_page.run()
