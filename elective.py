import streamlit as st
from elective_tab import eTab1, eTab2, eTab3
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("data/egovph.csv")
    return df

df=load_data()

df['at'] = pd.to_datetime(df['at'])
min_date = df['at'].min().date()
max_date = df['at'].max().date()

with st.sidebar:
    st.sidebar.title("Filters")
    st.markdown('**Date Filter**')
    start_date = st.date_input("Start Date", value=min_date)
    end_date = st.date_input("End Date", value=max_date)
    
    filtered_df = df[
    (df['at'] >= pd.to_datetime(start_date)) &
    (df['at'] <= pd.to_datetime(end_date))
    ]


# Page Setup
st.set_page_config(
    page_title="eGovPH Review Insight",
    page_icon=":philippines:",
    layout="wide",
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
    eTab1.render(filtered_df)
with u2:
    eTab2.render(filtered_df)
with u3:
    eTab3.render(filtered_df)
