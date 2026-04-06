import requests
import json
import re
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Check if stopwords are already downloaded before trying to download
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter

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
    
    # Collect all topics with their top words
    topics_data = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics_data.append({
            "topic_num": topic_idx + 1,
            "words": top_words
        })
        st.write(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")

    # ── AI-Generated Topic Labels ────────────────────────────────────────────────
    # Use OpenAI to generate understandable topic labels
    if st.button("Generate Topic Labels with AI"):
        with st.spinner("Analyzing topics..."):
            # Read API key from secrets.toml file directly
            try:
                with open('.streamlit/secrets.toml', 'r') as f:
                    secrets_content = f.read()
                    match = re.search(r'OPENROUTER_API_KEY\s*=\s*"([^"]+)"', secrets_content)
                    if match:
                        api_key = match.group(1)
                    else:
                        st.error("Could not find OPENROUTER_API_KEY in secrets.toml")
                        st.stop()
            except FileNotFoundError:
                st.error("Could not find .streamlit/secrets.toml file")
                st.stop()
            
            # Build the prompt with all topics
            topics_text = "\n".join([
                f"Topic {t['topic_num']}: {', '.join(t['words'])}"
                for t in topics_data
            ])
            
            prompt = f"""You are a data analyst. I have performed topic modelling on a dataset and got 10 topics. 
Below are the top 5 words for each topic. Please provide a short, understandable label (2-4 words) 
and a brief description (1-2 sentences) for each topic that would help a non-technical user understand what each topic represents.

{topics_text}

Format your response as:
Topic 1: [Label] - [Description]
Topic 2: [Label] - [Description]
and so on.
"""
            
            # Call the AI using requests directly
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "openai/gpt-oss-120b:free",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            
            if response.status_code != 200:
                st.error(f"API Error: {response.status_code} - {response.text}")
                st.stop()
            
            # Display the AI-generated labels
            result = response.json()
            st.subheader("AI-Generated Topic Labels")
            st.markdown(result['choices'][0]['message']['content'])

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
        width="stretch",
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