from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud # type: ignore
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter
import numpy as np

def render(df):
    st.header(":material/dictionary: Text Mining Analysis", anchor=False)

    # ── Stopwords ────────────────────────────────────────────────────────────
    STOPWORDS = set(stopwords.words('english'))
    APP_STOPWORDS = {
        "app", "apps", "application", "update", "version", "phone",
        "please", "use", "using", "used", "good", "great", "nice",
        "would", "could", "really", "like", "just", "get", "got",
        "one", "also", "even", "still", "now", "make", "made", "po",
        "opo", "ng", "cant", "sa", "u", "d", "na", "wow", "im", "ok", "app."
    }
    STOPWORDS.update(APP_STOPWORDS)

    # sklearn needs a list/frozenset, not a plain set, for stop_words param
    STOPWORDS_LIST = list(STOPWORDS)

    # ── Preprocessing ────────────────────────────────────────────────────────
    def preprocess_for_text_mining(text):
        words = str(text).split()
        words = [w for w in words if w.lower() not in STOPWORDS]
        return ' '.join(words)

    df['processed_text'] = df['translated'].apply(preprocess_for_text_mining)

    # ── Shared colour palette ─────────────────────────────────────────────────
    colors = ['#2c3e50', "#1136af", '#3498db', "#b92e2e", "#f00a0a"]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    # ── Topic Modelling (NMF) ────────────────────────────────────────────────
    st.subheader("Topic Modelling", anchor=False)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=STOPWORDS_LIST,   # ← consistent stopwords applied here
    )
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

    n_topics = 10
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_model.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")

    # ── Word Frequency Chart ─────────────────────────────────────────────────
    st.subheader("Top 20 Word Frequencies", anchor=False)
 
    all_tokens = [
        word.lower()
        for text in df['processed_text'].fillna('')
        for word in text.split()
        if word.lower() not in STOPWORDS and len(word) > 1
    ]
    freq = Counter(all_tokens).most_common(20)
    freq_df = pd.DataFrame(freq, columns=["word", "count"])
 
    st.vega_lite_chart(
        freq_df,
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": {"type": "bar", "cornerRadiusEnd": 4},
            "encoding": {
                "y": {
                    "field": "word",
                    "type": "nominal",
                    "sort": "-x",
                    "title": "Word",
                    "axis": {"labelFontSize": 12},
                },
                "x": {
                    "field": "count",
                    "type": "quantitative",
                    "title": "Frequency",
                    "axis": {"labelFontSize": 11},
                },
                "color": {
                    "field": "count",
                    "type": "quantitative",
                    "scale": {"range": colors},
                    "legend": None,
                },
                "tooltip": [
                    {"field": "word", "type": "nominal", "title": "Word"},
                    {"field": "count", "type": "quantitative", "title": "Frequency"},
                ],
            },
            "config": {"view": {"stroke": None}, "axis": {"grid": False}},
        },
        use_container_width=True,
    )
    
    # ── Word Cloud ────────────────────────────────────────────────────────────
    st.subheader("Word Cloud", anchor=False)

    all_words = ' '.join(df['processed_text'].fillna(''))
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        colormap=custom_cmap,
        stopwords=STOPWORDS,          # ← consistent stopwords applied here
    ).generate(all_words)

    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)