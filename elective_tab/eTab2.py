import streamlit as st
import streamlit_shadcn_ui as ui

def render(df):
    st.header(":material/dictionary: Text Mining Analysis",anchor=False)
    st.dataframe(df)