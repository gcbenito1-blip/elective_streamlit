import requests
import json
import re
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Check if stopwords and vader_lexicon are already downloaded before trying to download
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

def render(df):
    st.header(":material/cognition_2: Text Mining & Sentiment Analysis", anchor=False)
    
    # Initialize session state for topic labels and entity types if not exists
    if 'topic_labels' not in st.session_state:
        st.session_state.topic_labels = {}
    if 'entity_types' not in st.session_state:
        st.session_state.entity_types = {}
    if 'merged_topics' not in st.session_state:
        st.session_state.merged_topics = {}
    if 'split_topics' not in st.session_state:
        st.session_state.split_topics = {}
    
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
    
    # =========================================================================
    # SENTIMENT ANALYSIS SECTION (from eTab3)
    # =========================================================================
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
    
    # Sentiment over Time
    st.write("**Sentiment Over Time**")
    df_time = df.copy()
    df_time['at'] = pd.to_datetime(df_time['at'])
    df_time = df_time.sort_values('at')
    df_time['month'] = df_time['at'].dt.to_period('M').astype(str)
    monthly_stats = df_time.groupby('month').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        review_count=('sentiment_score', 'count'),
        positive_count=('sentiment', lambda x: (x == 'Positive').sum()),
        negative_count=('sentiment', lambda x: (x == 'Negative').sum()),
        neutral_count=('sentiment', lambda x: (x == 'Neutral').sum())
    ).reset_index()
    monthly_stats['positive_pct'] = (monthly_stats['positive_count'] / monthly_stats['review_count']) * 100
    monthly_stats['negative_pct'] = (monthly_stats['negative_count'] / monthly_stats['review_count']) * 100
    monthly_stats['neutral_pct'] = (monthly_stats['neutral_count'] / monthly_stats['review_count']) * 100
    
    sentiment_chart_data = pd.melt(
        monthly_stats, 
        id_vars=['month', 'review_count'], 
        value_vars=['positive_pct', 'neutral_pct', 'negative_pct'],
        var_name='sentiment_type', 
        value_name='percentage'
    )
    sentiment_map = {
        'positive_pct': ('Positive', '#2ecc71'),
        'neutral_pct': ('Neutral', '#95a5a6'), 
        'negative_pct': ('Negative', '#e74c3c')
    }
    sentiment_chart_data['sentiment_label'] = sentiment_chart_data['sentiment_type'].map(lambda x: sentiment_map[x][0])
    sentiment_chart_data['color'] = sentiment_chart_data['sentiment_type'].map(lambda x: sentiment_map[x][1])
    
    st.vega_lite_chart(
        sentiment_chart_data,
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": "container",
            "height": 400,
            "layer": [
                {
                    "mark": {
                        "type": "line",
                        "point": {"size": 50, "filled": True}
                    },
                    "encoding": {
                        "x": {
                            "field": "month",
                            "type": "temporal",
                            "title": "Month",
                            "axis": {"labelAngle": -45}
                        },
                        "y": {
                            "field": "percentage",
                            "type": "quantitative",
                            "title": "Sentiment Percentage (%)",
                            "scale": {"domain": [0, 100]},
                            "axis": {"labelFontSize": 11}
                        },
                        "color": {
                            "field": "sentiment_label",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Positive", "Neutral", "Negative"],
                                "range": ["#2ecc71", "#95a5a6", "#e74c3c"]
                            },
                            "legend": {"title": "Sentiment"}
                        },
                        "tooltip": [
                            {"field": "month", "type": "temporal", "title": "Month"},
                            {"field": "sentiment_label", "type": "nominal", "title": "Sentiment"},
                            {"field": "percentage", "type": "quantitative", "title": "Percentage", "format": ".1f"},
                            {"field": "review_count", "type": "quantitative", "title": "Review Volume"}
                        ]
                    }
                }
            ]
        }
    )
    
    # Sentiment Distribution
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("**Sentiment Distribution**")
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_df = pd.DataFrame({
            'Sentiment': sentiment_counts.index,
            'Count': sentiment_counts.values
        })
        st.vega_lite_chart(
            sentiment_df,
            {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "width": "container",
                "height": 400,
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
    
    with col2:
        st.write("**Score Distribution**")
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
                "height": 400,
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
                    "color": {"value": "#3498db"},
                    "tooltip": [
                        {"field": "bin_label", "type": "nominal", "title": "Score Range"},
                        {"field": "count", "type": "quantitative", "title": "Frequency"}
                    ],
                },
            }
        )
    
    # =========================================================================
    # TEXT MINING / TOPIC MODELING SECTION (from eTab2)
    # =========================================================================
    st.subheader("Text Mining & Topic Analysis", anchor=False)
    
    # Topic Modelling (NMF) on all reviews
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=STOPWORDS_LIST,
    )
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    
    n_topics = 10
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_model.fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    
    topics_data = []
    topic_document_matrix = nmf_model.transform(tfidf_matrix)
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        dominant_topic = np.argmax(topic_document_matrix, axis=1)
        prevalence = (dominant_topic == topic_idx).sum() / len(df) * 100
        topics_data.append({
            "topic_num": topic_idx + 1,
            "words": top_words,
            "prevalence": prevalence,
            "topic_idx": topic_idx
        })
    
    # Overview metrics
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        distinct_topics = len([t for t in topics_data if t['prevalence'] > 1])
        st.metric("Distinct Topics", distinct_topics)
    
    # Word Cloud
    st.write("**Word Cloud**")
    all_words = ' '.join(df['processed_text'].fillna(''))
    if all_words.strip():
        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=50,
            colormap=custom_cmap,
            stopwords=STOPWORDS,
        ).generate(all_words)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc, width="content")
    else:
        st.info("No text data available for word cloud")
    
    # Top 5 Topics Table
    st.write("**Top 5 Topics by Prevalence**")
    top_5_topics = sorted(topics_data, key=lambda x: x['prevalence'], reverse=True)[:5]
    if top_5_topics:
        overview_data = []
        for topic in top_5_topics:
            topic_num = topic['topic_num']
            label = st.session_state.topic_labels.get(topic_num, f"Topic {topic_num}")
            overview_data.append({
                "Topic": label,
                "Prevalence (%)": f"{topic['prevalence']:.1f}%",
                "Top Words": ", ".join(topic['words'][:5])
            })
        overview_df = pd.DataFrame(overview_data)
        st.dataframe(overview_df, width="stretch", hide_index=True)
    
    # Topic Explorer
    st.write("**Topic Explorer**")
    topic_options = {f"{st.session_state.topic_labels.get(t['topic_num'], f'Topic {t['topic_num']}')} ({t['prevalence']:.1f}%)": t['topic_num'] 
                     for t in topics_data if t['prevalence'] > 0.5}
    
    if topic_options:
        selected_topic_label = st.selectbox("Select Topic to Explore", options=list(topic_options.keys()), key="topic_explorer_select")
        selected_topic_num = topic_options[selected_topic_label]
        selected_topic_idx = selected_topic_num - 1
        
        topic_info = next(t for t in topics_data if t['topic_num'] == selected_topic_num)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Topic Words:** {', '.join(topic_info['words'])}")
            topic_doc_indices = np.where(np.argmax(topic_document_matrix, axis=1) == selected_topic_idx)[0]
            if len(topic_doc_indices) > 0:
                sample_size = min(5, len(topic_doc_indices))
                sample_indices = np.random.choice(topic_doc_indices, size=sample_size, replace=False)
                st.write("**Sample Excerpts:**")
                for idx in sample_indices:
                    excerpt = str(df.iloc[idx]['translated'])[:200] + "..." if len(str(df.iloc[idx]['translated'])) > 200 else str(df.iloc[idx]['translated'])
                    st.write(f"• {excerpt}")
            else:
                st.write("No documents strongly associated with this topic")
        
        with col2:
            st.metric("Prevalence", f"{topic_info['prevalence']:.1f}%", border=True)
            if len(topic_doc_indices) > 0:
                avg_score = df.iloc[topic_doc_indices]['sentiment_score'].mean()
                st.metric("Avg Sentiment", f"{avg_score:.2f}", border=True)
            else:
                st.metric("Avg Sentiment", "N/A", border=True)
            
            current_label = st.session_state.topic_labels.get(selected_topic_num, f"Topic {selected_topic_num}")
            new_label = st.text_input("Custom Label", value=current_label, key=f"label_{selected_topic_num}")
            if new_label != current_label:
                st.session_state.topic_labels[selected_topic_num] = new_label
                st.rerun()
    
    # Entity Extraction
    st.write("**Entity Extraction**")
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
    
    all_entities = []
    for text in df['translated']:
        entities = extract_entities(text)
        all_entities.extend(entities)
    
    entity_counts = Counter([entity for entity, _ in all_entities])
    entity_type_map = {entity: etype for entity, etype in all_entities}
    
    if entity_counts:
        entity_data = []
        for entity, count in entity_counts.most_common(10):
            entity_type = entity_type_map.get(entity, 'unknown')
            sample_reviews = []
            for idx, row in df.iterrows():
                if entity in str(row['translated']).lower():
                    sample_reviews.append(str(row['translated'])[:100] + "...")
                    if len(sample_reviews) >= 2:
                        break
            entity_data.append({
                "Entity": entity,
                "Type": entity_type.replace('_', ' ').title(),
                "Count": count,
                "Sample Reviews": " | ".join(sample_reviews) if sample_reviews else "No samples"
            })
        entity_df = pd.DataFrame(entity_data)
        st.dataframe(entity_df, width="stretch", hide_index=True)
    else:
        st.info("No entities detected in the current dataset")
    
    # Co-occurrence Analysis
    st.write("**Co-occurrence Analysis**")
    if 'entity_df' in locals() and len(entity_df) > 0 and len(top_5_topics) > 0:
        cooccur_data = []
        for topic in top_5_topics[:3]:
            topic_num = topic['topic_num']
            topic_label = st.session_state.topic_labels.get(topic_num, f"Topic {topic_num}")
            topic_word_set = set(topic['words'])
            for _, entity_row in entity_df.head(5).iterrows():
                entity = entity_row['Entity']
                cooccur_score = 1 if entity in topic_word_set or any(word in entity.lower() for word in topic['words']) else 0
                if cooccur_score > 0:
                    cooccur_data.append({
                        "Topic": topic_label,
                        "Entity": entity,
                        "Co-occurrence Score": cooccur_score
                    })
        if cooccur_data:
            cooccur_df = pd.DataFrame(cooccur_data)
            st.dataframe(cooccur_df, width="stretch", hide_index=True)
        else:
            st.info("No significant co-occurrences found between top topics and entities")
    else:
        st.info("Insufficient data for co-occurrence analysis")
    
