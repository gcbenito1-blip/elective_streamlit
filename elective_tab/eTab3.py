import requests
import json
import re
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter
import numpy as np

# Check if stopwords and vader_lexicon are already downloaded before trying to download
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

def render(df):
    st.header("Negative Reviews Deep Dive", anchor=False)
    
    # Initialize session state for topic labels if not exists
    if 'topic_labels' not in st.session_state:
        st.session_state.topic_labels = {}
    
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
    
    # ── SENTIMENT ANALYSIS (required for negative reviews) ──────────────────
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
    
    # Apply sentiment analysis if not already present
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df['translated'].apply(get_sentiment_scores)
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)
    
    # =========================================================================
    # NEGATIVE REVIEWS ANALYSIS
    # =========================================================================
    negative_df = df[df['sentiment'] == 'Negative'][['translated', 'sentiment_score', 'processed_text']].copy()
    negative_df = negative_df.reset_index().rename(columns={'index': 'original_index'})
    
    if len(negative_df) > 0:
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
    
    if len(negative_df) > 0:
        # Build negative topics data
        negative_topics_data = []
        for topic_idx in negative_df['topic'].unique():
            if topic_idx == -1:
                continue
            topic_reviews = negative_df[negative_df['topic'] == topic_idx]
            top_words = [feature_names_neg[i] for i in nmf_neg.components_[topic_idx].argsort()[:-6:-1]]
            negative_topics_data.append({
                "topic_num": int(topic_idx),
                "words": top_words,
                "prevalence": len(topic_reviews) / len(negative_df) * 100,
                "count": len(topic_reviews),
                "avg_sentiment": topic_reviews['sentiment_score'].mean()
            })
        
        # Sort by count descending
        negative_topics_data.sort(key=lambda x: x['count'], reverse=True)
        
        # Overview metrics
        col1, col2, col3 = st.columns(3, border=True)
        with col1:
            st.metric("Total Negative Reviews", len(negative_df))
        with col2:
            distinct_neg_topics = len([t for t in negative_topics_data if t['prevalence'] > 0])
            st.metric("Negative Topics Found", distinct_neg_topics)
        with col3:
            avg_neg_sentiment = negative_df['sentiment_score'].mean()
            st.metric("Avg Negative Sentiment", f"{avg_neg_sentiment:.2f}")
        
        # Topic Distribution Chart
        st.write("**Topic Distribution in Negative Reviews**")
        topic_dist = negative_df['topic_label'].value_counts()
        if len(topic_dist) > 0:
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
                            "title": "Topic Keywords",
                            "axis": {"labelFontSize": 11},
                        },
                        "x": {
                            "field": "Count",
                            "type": "quantitative",
                            "title": "Count",
                            "axis": {"labelFontSize": 11},
                        },
                        "color": {"value": "#e74c3c"},
                        "tooltip": [
                            {"field": "Topic", "type": "nominal", "title": "Topic"},
                            {"field": "Count", "type": "quantitative", "title": "Count"}
                        ],
                    },
                }
            )
            
            # AI-Generated Topic Labels for Negative Reviews
            st.write("**AI-Generated Topic Labels**")
            if st.button("Generate Negative Topic Labels with AI", key="ai_neg_labels_btn"):
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
                    
                    topics_text = "\n".join([
                        f"Topic {i+1} (keywords: {', '.join(t['words'])})" 
                        for i, t in enumerate(negative_topics_data)
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
                        
                        result = response.json()
                        st.subheader("AI-Generated Negative Review Topic Labels")
                        st.markdown(result['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"Error generating AI labels: {str(e)}")
                        st.stop()
        # Topic Explorer for Negative Reviews
        st.write("**Topic Explorer - Click on a topic to explore reviews**")
        
        topic_options = {f"{st.session_state.topic_labels.get(t['topic_num'], f'Topic {t['topic_num']}')} ({t['count']} reviews, {t['prevalence']:.1f}%)": t['topic_num'] 
                         for t in negative_topics_data}
        
        if topic_options:
            selected_topic_label = st.selectbox("Select Negative Topic to Explore", options=list(topic_options.keys()), key="neg_topic_explorer_select")
            selected_topic_num = topic_options[selected_topic_label]
            selected_topic_idx = selected_topic_num
            
            topic_info = next(t for t in negative_topics_data if t['topic_num'] == selected_topic_num)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**Topic Keywords:** {', '.join(topic_info['words'])}")
            with col2:
                st.metric("Review Count", topic_info['count'], border=True)
            with col3:
                st.metric("Avg Sentiment", f"{topic_info['avg_sentiment']:.2f}", border=True)
            
            # Get sample reviews for this topic
            topic_reviews = negative_df[negative_df['topic'] == selected_topic_num].copy()
            topic_reviews = topic_reviews.sort_values('sentiment_score')

            topic1, topic2 = st.columns(2) 
            with topic1:
                st.write("**Sample Reviews from This Topic:**")
                sample_size = min(10, len(topic_reviews))
                for idx, row in topic_reviews.head(sample_size).iterrows():
                    with st.container(border=True):
                        st.markdown(f"**Score:** {row['sentiment_score']:.2f} :material/star:")
                        st.markdown(f"{row['translated']}")

            # Entity Extraction on Negative Reviews
            with topic2:
                st.write("**Entity Extraction - Negative Reviews**")
                def extract_entities(text):
                    text_lower = str(text).lower()
                    entities = []
                    feature_keywords = ['login', 'password', 'crash', 'slow', 'fast', 'ui', 'interface', 
                                    'button', 'menu', 'notification', 'update', 'feature', 'function']
                    os_keywords = ['android', 'samsung', 'xiaomi', 'oppo', 'vivo', 
                                'tablet', 'phone', 'device']
                    for keyword in feature_keywords:
                        if keyword in text_lower:
                            entities.append((keyword, 'feature'))
                    for keyword in os_keywords:
                        if keyword in text_lower:
                            entities.append((keyword, 'os_device'))
                    seen = set()
                    unique_entities = []
                    for entity, etype in entities:
                        if (entity, etype) not in seen:
                            seen.add((entity, etype))
                            unique_entities.append((entity, etype))
                    return unique_entities[:5]
                
                all_entities_neg = []
                for text in negative_df['translated']:
                    entities = extract_entities(text)
                    all_entities_neg.extend(entities)
                
                entity_counts_neg = Counter([entity for entity, _ in all_entities_neg])
                entity_type_map_neg = {entity: etype for entity, etype in all_entities_neg}
                
                if entity_counts_neg:
                    entity_data_neg = []
                    for entity, count in entity_counts_neg.most_common(10):
                        entity_type = entity_type_map_neg.get(entity, 'unknown')
                        sample_reviews = []
                        for idx, row in negative_df.iterrows():
                            if entity in str(row['translated']).lower():
                                sample_reviews.append(str(row['translated'])[:100] + "...")
                                if len(sample_reviews) >= 2:
                                    break
                        entity_data_neg.append({
                            "Entity": entity,
                            "Type": entity_type.replace('_', ' ').title(),
                            "Count": count,
                            "Sample Reviews": " | ".join(sample_reviews) if sample_reviews else "No samples"
                        })
                    entity_df_neg = pd.DataFrame(entity_data_neg)
                    st.dataframe(entity_df_neg, width="stretch", hide_index=True)
                else:
                    st.info("No entities detected in the negative reviews")
                
                # Co-occurrence Analysis for Negative Reviews
                st.write("**Co-occurrence Analysis - Negative Reviews**")
                if 'entity_df_neg' in locals() and len(entity_df_neg) > 0 and len(negative_topics_data) > 0:
                    cooccur_data_neg = []
                    for topic in negative_topics_data[:3]:
                        topic_num = topic['topic_num']
                        topic_label = st.session_state.topic_labels.get(topic_num, f"Topic {topic_num}")
                        topic_word_set = set(topic['words'])
                        for _, entity_row in entity_df_neg.head(5).iterrows():
                            entity = entity_row['Entity']
                            cooccur_score = 1 if entity in topic_word_set or any(word in entity.lower() for word in topic['words']) else 0
                            if cooccur_score > 0:
                                cooccur_data_neg.append({
                                    "Topic": topic_label,
                                    "Entity": entity,
                                    "Co-occurrence Score": cooccur_score
                                })
                    if cooccur_data_neg:
                        cooccur_df_neg = pd.DataFrame(cooccur_data_neg)
                        st.dataframe(cooccur_df_neg, width="stretch", hide_index=True)
                    else:
                        st.info("No significant co-occurrences found between negative topics and entities")
                else:
                    st.info("Insufficient data for co-occurrence analysis")