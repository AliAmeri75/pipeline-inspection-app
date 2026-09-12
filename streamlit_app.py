#!/usr/bin/env python3
"""Application entry point and page navigation."""

import streamlit as st


st.set_page_config(
    page_title="Pipeline Inspection Planning",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

introduction = st.Page(
    "introduction.py",
    title="Introduction",
    icon=":material/home:",
    default=True,
)
inspection_planner = st.Page(
    "inspection_planner.py",
    title="Inspection scheduling",
    icon=":material/timeline:",
)

navigation = st.navigation([introduction, inspection_planner])
navigation.run()
