
import streamlit as st
import pandas as pd


def render(df):
    st.header(":material/cognition_2: Sentiment Analysis", anchor=False)

    st.dataframe(df)