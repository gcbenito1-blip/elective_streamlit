
import streamlit as st
import pandas as pd


def render(df):
    st.header(":material/cognition_2: Sentiment Analysis", anchor=False)
    # Example lexicon
    lexicon = {'good': 1, 'great': 1, 'excellent': 1, 'bad': -1, 'poor': -1, 'terrible': -1}

    def lexicon_sentiment(text):
        score = sum([lexicon.get(word.lower(), 0) for word in text.split()])
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        else:
            return 'neutral'

    df['lexicon_sentiment'] = df['translated'].apply(lexicon_sentiment)
    st.dataframe(df)