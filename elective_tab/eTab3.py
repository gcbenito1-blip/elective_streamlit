import streamlit as st
import requests
import json
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Check if stopwords are already downloaded before trying to download
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ── Stopwords ───────────────────────────────────────────────────────────────────
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


def render(df):
    st.header(":material/cognition_2: Sentiment Analysis", anchor=False)

    # ── Preprocessing ───────────────────────────────────────────────────────────
    def preprocess_for_text_mining(text):
        words = str(text).split()
        words = [w for w in words if w.lower() not in STOPWORDS]
        return ' '.join(words)

    df['processed_text'] = df['translated'].apply(preprocess_for_text_mining)

    # ── Sentiment Analysis Pipeline ──────────────────────────────────────────────
    st.subheader("Sentiment Analysis", anchor=False)

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    def get_sentiment_scores(text):
        """Get compound sentiment score using VADER."""
        if pd.isna(text) or str(text).strip() == '':
            return 0.0
        return sia.polarity_scores(str(text))['compound']

    def classify_sentiment(score):
        """Classify sentiment based on compound score."""
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply sentiment analysis
    df['sentiment_score'] = df['translated'].apply(get_sentiment_scores)
    df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

    # ── Sentiment Distribution Overview ─────────────────────────────────────────
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Positive", f"{sentiment_counts.get('Positive', 0)}", 
                  )
    with col2:
        st.metric("Neutral", f"{sentiment_counts.get('Neutral', 0)}", 
                  )
    with col3:
        st.metric("Negative", f"{sentiment_counts.get('Negative', 0)}", 
                  )



    # ── Sentiment Distribution (Vega-Lite) ───────────────────────────────────────
    st.subheader("Sentiment Distribution", anchor=False)

    sentiment_df = pd.DataFrame({
        'Sentiment': sentiment_counts.index,
        'Count': sentiment_counts.values
    })

    st.vega_lite_chart(
        sentiment_df,
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": "container",
            "height": 480,
            "mark": {"type": "bar", "cornerRadiusEnd": 4},
            "encoding": {
                "x": {
                    "field": "Sentiment",
                    "type": "nominal",
                    "title": "Sentiment",
                    "axis": {"labelFontSize": 12},
                },
                "y": {
                    "field": "Count",
                    "type": "quantitative",
                    "title": "Count",
                    "axis": {"labelFontSize": 11},
                },
                "color": {
                    "field": "Sentiment",
                    "type": "nominal",
                    "scale": {
                        "domain": ["Positive", "Neutral", "Negative"],
                        "range": ["#2ecc71", "#95a5a6", "#e74c3c"]
                    },
                    "legend": None,
                },
                "tooltip": [
                    {"field": "Sentiment", "type": "nominal", "title": "Sentiment"},
                    {"field": "Count", "type": "quantitative", "title": "Count"}
                ],
            },
        }
    )

    # ── Sentiment Score Distribution (Vega-Lite) ─────────────────────────────────
    st.subheader("Sentiment Score Distribution", anchor=False)

    # Create bins for histogram
    score_bins = pd.cut(df['sentiment_score'], bins=20)
    score_dist = score_bins.value_counts().sort_index()
    score_hist_df = pd.DataFrame({
        'bin_start': [idx.left for idx in score_dist.index],
        'bin_end': [idx.right for idx in score_dist.index],
        'count': score_dist.values
    })
    score_hist_df['bin_label'] = score_hist_df['bin_start'].round(2).astype(str) + ' to ' + score_hist_df['bin_end'].round(2).astype(str)

    st.vega_lite_chart(
        score_hist_df,
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": "container",
            "height": 480,
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "x": {
                    "field": "bin_label",
                    "type": "nominal",
                    "title": "Sentiment Score Range",
                    "axis": {"labelAngle": -45, "labelFontSize": 10},
                },
                "y": {
                    "field": "count",
                    "type": "quantitative",
                    "title": "Frequency",
                    "axis": {"labelFontSize": 11},
                },
                "color": {
                    "value": "#3498db"
                },
                "tooltip": [
                    {"field": "bin_label", "type": "nominal", "title": "Score Range"},
                    {"field": "count", "type": "quantitative", "title": "Frequency"}
                ],
            },
        }
    )

    # ── Negative Sentiments with Topics ────────────────────────────────────────
    st.subheader("Negative Reviews with Topics", anchor=False)

    # Get negative reviews
    negative_df = df[df['sentiment'] == 'Negative'][['translated', 'sentiment_score', 'processed_text']].copy()
    negative_df = negative_df.reset_index()
    negative_df = negative_df.rename(columns={'index': 'original_index'})

    if len(negative_df) > 0:
        # Topic modeling on negative reviews
        vectorizer_neg = TfidfVectorizer(
            max_features=500,
            stop_words=STOPWORDS_LIST,
        )
        try:
            tfidf_neg = vectorizer_neg.fit_transform(negative_df['processed_text'].fillna(''))
            n_topics_neg = min(10, tfidf_neg.shape[0], tfidf_neg.shape[1])
            if n_topics_neg > 0:
                nmf_neg = NMF(n_components=n_topics_neg, random_state=42)
                nmf_neg.fit(tfidf_neg)
                feature_names_neg = vectorizer_neg.get_feature_names_out()
                
                # Assign topic to each negative review
                topic_matrix = nmf_neg.transform(tfidf_neg)
                negative_df['topic'] = topic_matrix.argmax(axis=1)
                negative_df['topic_label'] = negative_df['topic'].apply(
                    lambda x: ', '.join([feature_names_neg[i] for i in nmf_neg.components_[x].argsort()[:-4:-1]])
                )
            else:
                negative_df['topic'] = -1
                negative_df['topic_label'] = 'N/A'
        except Exception:
            negative_df['topic'] = -1
            negative_df['topic_label'] = 'N/A'
    else:
        st.info("No negative reviews found.")

    # Display negative reviews dataframe
    if len(negative_df) > 0:
        display_cols = ['translated', 'sentiment_score', 'topic_label']
        st.dataframe(
            negative_df[display_cols].sort_values('sentiment_score'),
            width="stretch",
            height=480
        )
        
        # Topic distribution for negative reviews (Vega-Lite bar chart)
        topic_dist = negative_df['topic_label'].value_counts()
        if len(topic_dist) > 0:
            st.subheader("Topics in Negative Reviews", anchor=False)
            topic_df = pd.DataFrame({'Topic': topic_dist.index, 'Count': topic_dist.values})

            st.vega_lite_chart(
                topic_df,
                {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "width": "container",
                    "height": 450,
                    "mark": {"type": "bar", "cornerRadiusEnd": 4},
                    "encoding": {
                        "y": {
                            "field": "Topic",
                            "type": "nominal",
                            "sort": "-x",
                            "title": "Topic",
                            "axis": {"labelFontSize": 11},
                        },
                        "x": {
                            "field": "Count",
                            "type": "quantitative",
                            "title": "Count",
                            "axis": {"labelFontSize": 11},
                        },
                        "color": {
                            "value": "#e74c3c"
                        },
                        "tooltip": [
                            {"field": "Topic", "type": "nominal", "title": "Topic"},
                            {"field": "Count", "type": "quantitative", "title": "Count"}
                        ],
                    },
                }
            )

            # ── AI-Generated Topic Labels for Negative Reviews ─────────────────────
            # Use OpenAI to generate understandable topic labels
            if st.button("Generate Negative Topic Labels with AI"):
                with st.spinner("Analyzing negative review topics..."):
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
                    except Exception as e:
                        st.error(f"Error reading API key: {str(e)}")
                        st.stop()
                    
                    # Build the prompt with all topics
                    topics_text = "\n".join([
                        f"Topic: {topic}" for topic in topic_dist.index
                    ])
                    
                    prompt = f"""You are a data analyst. I have performed topic modelling on negative reviews from an app store.
Below are the top keywords/phrases for each topic identified in the negative reviews. Please provide a short, understandable label (2-4 words) 
and a brief description (1-2 sentences) for each topic that would help a non-technical user understand what customers are complaining about.

{topics_text}

Format your response as:
Topic: [Label] - [Description]
and so on.
"""
                    
                    try:
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
                        st.subheader("AI-Generated Negative Review Topic Labels")
                        st.markdown(result['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"Error generating AI labels: {str(e)}")
                        st.stop()

