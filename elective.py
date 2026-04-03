import streamlit as st
from elective_tab import eTab1, eTab2, eTab3
import pandas as pd

with st.sidebar:
    st.sidebar.title("Filters")
    with st.container(border=True, ):
        st.checkbox(label="Pls select", value="s")
@st.cache_data
def load_data():
    df = pd.read_csv("data/egovph.csv")
    return df


df=load_data()

# Page Setup
st.set_page_config(
    page_title="eGovPH Review Insight",
    page_icon=":philippines:",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Text Mining and Sentiment Analysis"
    }
)

st.markdown(
    """
    <style>
    .sub-text {
        color: grey;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.title(":material/bar_chart: eGovPH Review Insight", anchor=False)
st.markdown("<p class='sub-text'>Text Mining and Sentiment analysis for eGovPH Android App</p>", unsafe_allow_html=True)

with st.container():
    u1, u2, u3 = st.tabs([':material/dashboard: Overview', ':material/dictionary: Text Mining Analysis', ':material/cognition_2: Sentiment Analysis'])

with u1:
    eTab1.render(df)
with u2:
    eTab2.render(df)
with u3:
    eTab3.render(df)
