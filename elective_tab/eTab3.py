
import streamlit as st
import streamlit_shadcn_ui as ui

def render(df):
    st.header(":material/cognition_2: Sentiment Analysis", anchor=False)
    st.dataframe(df)