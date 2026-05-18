import pandas as pd
from google_play_scraper import reviews, Sort
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Download VADER lexicon if not already present
@st.cache_resource
def download_vader_lexicon():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

# Initialize VADER
download_vader_lexicon()

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching reviews…")
def scrape_data(app_id: str, count: int):
    result, _ = reviews(app_id, lang="en", sort=Sort.NEWEST, count=count)
    return result

def get_sentiment_vader(text):
    """
    Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)
    Returns sentiment label and compound score
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    
    # VADER thresholds (adjustable based on needs)
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Return sentiment label and all scores for reference
    return sentiment, round(compound, 3), scores

def clean_text(text):
    text = re.sub(r"http\S+|[^a-z\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()

STOPWORDS = {
    "the","and","is","it","in","of","to","a","this","for","that","was","i","on",
    "are","with","have","but","not","they","so","we","be","an","at","as","my",
    "me","you","app","apps","egov","gov","government","its","or","do","if","can",
    "has","by","use","used","get","got","just","would","could","should","very",
    "all","from","will","your","our","had","been","were","what","when","how",
    "one","out","no","up","im","ive","dont","cant","wont","didnt","doesnt",
}

def make_wordcloud(texts, colormap="Blues"):
    words = [w for w in " ".join(texts).split() if w not in STOPWORDS and len(w) > 2]
    if not words:
        return None
    wc = WordCloud(width=700, height=280, background_color="white",
                   colormap=colormap, max_words=80, collocations=False
                   ).generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

# ── Render function (call this inside your tab) ───────────────────────────────
def render(app_id: str = "egov.app", count: int = 500):
    # --- controls
    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl:
        count = st.slider("Reviews to load", 100, 1000, count, 100, key="rv_count")
    with col_btn:
        st.write("")
        refresh = st.button("🔄 Refresh", key="rv_refresh", width='content')
    
    cache_key = f"rv_df_{app_id}"
    if cache_key not in st.session_state or refresh:
        with st.spinner("Scraping and analyzing reviews..."):
            raw = scrape_data(app_id, count)
            df = pd.DataFrame(raw)
            df["at"] = pd.to_datetime(df["at"])
            # Use VADER for sentiment analysis
            sentiment_data = df["content"].apply(get_sentiment_vader)
            df["sentiment"] = sentiment_data.apply(lambda x: x[0])
            df["polarity"] = sentiment_data.apply(lambda x: x[1])
            df["vader_scores"] = sentiment_data.apply(lambda x: x[2])
            df["clean"] = df["content"].map(clean_text)
            st.session_state[cache_key] = df

    df = st.session_state[cache_key]

    # --- KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Reviews", f"{len(df):,}")
    k2.metric("Avg Rating", f"{df['score'].mean():.2f} ⭐")
    k3.metric("😊 Positive", f"{(df['sentiment']=='Positive').mean()*100:.1f}%")
    k4.metric("😐 Neutral",  f"{(df['sentiment']=='Neutral').mean()*100:.1f}%")
    k5.metric("😠 Negative", f"{(df['sentiment']=='Negative').mean()*100:.1f}%")

    st.divider()

    # --- Charts row
    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("**Sentiment Distribution (VADER)**")
        counts = df["sentiment"].value_counts().reindex(["Positive","Neutral","Negative"])
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.bar(counts.index, counts.values, color=["#2ecc71","#95a5a6","#e74c3c"], width=0.5)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0, len(df))
        fig.tight_layout()
        st.pyplot(fig)

    with c2:
        st.caption("**Ratings**")
        rc = df["score"].value_counts().sort_index()
        colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"]
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.barh([str(i)+"★" for i in rc.index], rc.values,
                color=[colors[i-1] for i in rc.index])
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    with c3:
        st.caption("**Top Keywords**")
        words = [w for w in " ".join(df["clean"]).split()
                 if w not in STOPWORDS and len(w) > 2]
        top = Counter(words).most_common(8)
        wd, wc = zip(*top) if top else ([], [])
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.barh(list(wd)[::-1], list(wc)[::-1], color="#3498db")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    st.divider()

    # --- Word cloud tabs
    st.caption("**Word Cloud**")
    t1, t2, t3 = st.tabs(["All", "😊 Positive", "😠 Negative"])
    with t1:
        fig = make_wordcloud(df["clean"], "Blues")
        if fig: st.pyplot(fig)
    with t2:
        fig = make_wordcloud(df[df["sentiment"]=="Positive"]["clean"], "Greens")
        if fig: st.pyplot(fig)
        else: st.info("Not enough positive reviews.")
    with t3:
        fig = make_wordcloud(df[df["sentiment"]=="Negative"]["clean"], "Reds")
        if fig: st.pyplot(fig)
        else: st.info("Not enough negative reviews.")

    st.divider()

    # --- Sample reviews with VADER score details
    st.caption("**Sample Reviews with VADER Analysis**")
    sent_filter = st.selectbox("Filter", ["All","Positive","Neutral","Negative"],
                               key="rv_filter", label_visibility="collapsed")
    view = df if sent_filter == "All" else df[df["sentiment"] == sent_filter]
    
    # Prepare display dataframe with VADER details
    display_df = view[["at","score","sentiment","polarity","content"]].copy()
    display_df["VADER Score"] = display_df["polarity"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(
        display_df[["at","score","sentiment","VADER Score","content"]]
        .rename(columns={"at":"Date","score":"★","sentiment":"Sentiment",
                         "content":"Review"})
        .sort_values("Date", ascending=False)
        .head(50)
        .reset_index(drop=True),
        width='content',
        height=250,
    )
    