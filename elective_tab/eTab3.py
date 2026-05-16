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
import hashlib
import time

# Check if stopwords and vader_lexicon are already downloaded before trying to download
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

@st.cache_data(ttl=3600, show_spinner="Generating AI complaint insights...")
def generate_negative_ai_topic_labels_cached(topics_hash, topics_data, api_key):
    """Generate AI-based topic labels for negative reviews with caching"""
    topics_text = "\n".join([
        f"Topic {i+1} (keywords: {', '.join(t['words'][:8])})" 
        for i, t in enumerate(topics_data)
    ])
    
    prompt = f"""You are a data analyst specializing in customer feedback. I have performed topic modelling on NEGATIVE reviews from an app store.
Below are the top keywords/phrases for each complaint topic identified. Please provide a short, actionable label (2-4 words) 
and a brief description (1-2 sentences) for each topic that explains WHAT users are complaining about and WHY.

{topics_text}

Format your response EXACTLY as:
Topic 1: [Label] - [Description]
Topic 2: [Label] - [Description]
and so on. Focus on the core issue users are facing.
"""
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful data analyst specializing in negative app review analysis."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }),
            timeout=30
        )
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            return None
        
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating AI labels: {str(e)}")
        return None

def parse_negative_ai_labels(ai_response, topics_data):
    """Parse AI response and extract topic labels for negative reviews"""
    labels = {}
    lines = ai_response.strip().split('\n')
    
    for i, topic in enumerate(topics_data):
        topic_num = i + 1
        for line in lines:
            match = re.match(rf'Topic\s*{topic_num}\s*:\s*([^-]+?)\s*-\s*(.+)', line, re.IGNORECASE)
            if match:
                label = match.group(1).strip()
                description = match.group(2).strip()
                labels[topic_num] = {'label': label, 'description': description}
                break
        
        if topic_num not in labels:
            labels[topic_num] = {
                'label': f"Complaint {topic_num}",
                'description': f"Issues related to: {', '.join(topic['words'][:3])}"
            }
    
    return labels

def get_api_key():
    """Retrieve API key from secrets.toml with caching"""
    try:
        with open('.streamlit/secrets.toml', 'r') as f:
            secrets_content = f.read()
            match = re.search(r'OPENROUTER_API_KEY\s*=\s*"([^"]+)"', secrets_content)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass
    return None

def render(df):
    st.header(":material/warning: Negative Reviews Deep Dive", anchor=False)
    
    # Initialize session state for topic labels if not exists
    if 'negative_topic_labels' not in st.session_state:
        st.session_state.negative_topic_labels = {}
    if 'negative_topic_descriptions' not in st.session_state:
        st.session_state.negative_topic_descriptions = {}
    if 'negative_topics_generated' not in st.session_state:
        st.session_state.negative_topics_generated = False
    if 'negative_topics_hash' not in st.session_state:
        st.session_state.negative_topics_hash = None
    
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
        if pd.isna(text) or str(text).strip() == '':
            return 0.0
        return sia.polarity_scores(str(text))['compound']
    
    def classify_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df['translated'].apply(get_sentiment_scores)
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)
    
    # =========================================================================
    # NEGATIVE REVIEWS ANALYSIS
    # =========================================================================
    negative_df = df[df['sentiment'] == 'Negative'][['translated', 'sentiment_score', 'processed_text']].copy()
    negative_df = negative_df.reset_index().rename(columns={'index': 'original_index'})
    
    if len(negative_df) == 0:
        st.info("🎉 No negative reviews found! Users are happy with the app.")
        return
    
    # Topic modelling on negative reviews
    with st.spinner("Analyzing complaint patterns..."):
        vectorizer_neg = TfidfVectorizer(
            max_features=500,
            stop_words=STOPWORDS_LIST,
        )
        tfidf_neg = vectorizer_neg.fit_transform(negative_df['processed_text'].fillna(''))
        n_topics_neg = min(8, tfidf_neg.shape[0], tfidf_neg.shape[1])
        
        if n_topics_neg > 0:
            nmf_neg = NMF(n_components=n_topics_neg, random_state=42)
            nmf_neg.fit(tfidf_neg)
            feature_names_neg = vectorizer_neg.get_feature_names_out()
            topic_matrix = nmf_neg.transform(tfidf_neg)
            negative_df['topic'] = topic_matrix.argmax(axis=1)
            
            # Build negative topics data
            negative_topics_data = []
            for topic_idx in negative_df['topic'].unique():
                topic_reviews = negative_df[negative_df['topic'] == topic_idx]
                top_words = [feature_names_neg[i] for i in nmf_neg.components_[topic_idx].argsort()[:-6:-1]]
                negative_topics_data.append({
                    "topic_num": int(topic_idx) + 1,
                    "words": top_words,
                    "prevalence": len(topic_reviews) / len(negative_df) * 100,
                    "count": len(topic_reviews),
                    "avg_sentiment": topic_reviews['sentiment_score'].mean()
                })
            
            # Sort by count descending
            negative_topics_data.sort(key=lambda x: x['count'], reverse=True)
            
            # Auto-generate AI Topic Labels (cached)
            api_key = get_api_key()
            
            if api_key and not st.session_state.negative_topics_generated:
                # Create a hash of the topics data to use as cache key
                current_hash = hashlib.md5(str(negative_topics_data).encode()).hexdigest()
                st.session_state.negative_topics_hash = current_hash
                
                ai_response = generate_negative_ai_topic_labels_cached(current_hash, negative_topics_data, api_key)
                
                if ai_response:
                    ai_labels = parse_negative_ai_labels(ai_response, negative_topics_data)
                    for topic_num, label_info in ai_labels.items():
                        st.session_state.negative_topic_labels[topic_num] = label_info['label']
                        st.session_state.negative_topic_descriptions[topic_num] = label_info['description']
                    st.session_state.negative_topics_generated = True
                else:
                    # Fallback to default labels
                    for topic in negative_topics_data:
                        topic_num = topic['topic_num']
                        st.session_state.negative_topic_labels[topic_num] = f"Complaint {topic_num}"
                        st.session_state.negative_topic_descriptions[topic_num] = f"Issues related to: {', '.join(topic['words'][:3])}"
                    st.session_state.negative_topics_generated = True
            elif not api_key:
                # Set default labels if no API key
                for topic in negative_topics_data:
                    topic_num = topic['topic_num']
                    if topic_num not in st.session_state.negative_topic_labels:
                        st.session_state.negative_topic_labels[topic_num] = f"Complaint {topic_num}"
                        st.session_state.negative_topic_descriptions[topic_num] = f"Issues related to: {', '.join(topic['words'][:3])}"
                st.session_state.negative_topics_generated = True
                
                # Show warning about missing API key
                warning_placeholder = st.empty()
                warning_placeholder.warning("⚠️ OpenRouter API key not found. Using default labels. Add OPENROUTER_API_KEY to .streamlit/secrets.toml for AI-powered insights.")
                time.sleep(3)
                warning_placeholder.empty()
            
            # Show success message briefly
            if st.session_state.negative_topics_generated and api_key:
                success_placeholder = st.empty()
                success_placeholder.success("✅ AI complaint insights loaded!")
                time.sleep(2)
                success_placeholder.empty()
        else:
            st.warning("Not enough negative reviews for detailed topic analysis.")
            return
    
    # Overview metrics
    col1, col2, col3 = st.columns(3, border=True)
    with col1:
        st.metric("Total Negative Reviews", len(negative_df))
    with col2:
        st.metric("Complaint Categories", len(negative_topics_data))
    with col3:
        avg_neg_sentiment = negative_df['sentiment_score'].mean()
        st.metric("Avg Severity", f"{avg_neg_sentiment:.2f}", help="Lower = more severe complaints")
    
    # Topic Distribution Chart with AI Labels
    st.write("**Complaint Distribution**")
    
    # Create distribution data with AI labels
    dist_data = []
    for topic in negative_topics_data:
        label = st.session_state.negative_topic_labels.get(topic['topic_num'], f"Topic {topic['topic_num']}")
        dist_data.append({
            "Complaint Type": label,
            "Count": topic['count']
        })
    
    topic_dist_df = pd.DataFrame(dist_data)
    st.vega_lite_chart(
        topic_dist_df,
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": "container",
            "height": 450,
            "mark": {"type": "bar", "cornerRadiusEnd": 4},
            "encoding": {
                "y": {
                    "field": "Complaint Type",
                    "type": "nominal",
                    "sort": "-x",
                    "title": "Complaint Category",
                    "axis": {"labelFontSize": 11},
                },
                "x": {
                    "field": "Count",
                    "type": "quantitative",
                    "title": "Number of Complaints",
                    "axis": {"labelFontSize": 11},
                },
                "color": {"value": "#e74c3c"},
                "tooltip": [
                    {"field": "Complaint Type", "type": "nominal", "title": "Issue"},
                    {"field": "Count", "type": "quantitative", "title": "Count"}
                ]
            },
        }
    )
    
    # Detailed Topics Table
    st.write("**Complaint Categories Details**")
    topic_details = []
    for topic in negative_topics_data:
        label = st.session_state.negative_topic_labels.get(topic['topic_num'], f"Category {topic['topic_num']}")
        description = st.session_state.negative_topic_descriptions.get(topic['topic_num'], "AI insights loading...")
        topic_details.append({
            "Issue": label,
            "Description": description[:150] + "..." if len(description) > 150 else description,
            "Occurrences": topic['count'],
            "Percentage": f"{topic['prevalence']:.1f}%",
            "Key Indicators": ", ".join(topic['words'][:4])
        })
    
    st.dataframe(pd.DataFrame(topic_details), width='content', hide_index=True)

    if st.button("🔄 Regenerate AI Labels", key="regenerate_neg_ai_labels", help="Regenerate AI complaint labels"):
        # Clear cache and reset
        st.cache_data.clear()
        st.session_state.negative_topics_generated = False
        st.session_state.negative_topic_labels.clear()
        st.session_state.negative_topic_descriptions.clear()
        st.rerun()
    
    # Topic Explorer for Negative Reviews
    st.write("**Complaint Explorer - Select an issue to see examples**")
    
    topic_options = {}
    for t in negative_topics_data:
        label = st.session_state.negative_topic_labels.get(t['topic_num'], f"Category {t['topic_num']}")
        topic_options[f"{label} ({t['count']} complaints, {t['prevalence']:.0f}%)"] = t['topic_num']
    
    if topic_options:
        selected_topic_label = st.selectbox("Select Complaint Type", options=list(topic_options.keys()), key="neg_topic_explorer_select")
        selected_topic_num = topic_options[selected_topic_label]
        
        # Find the actual topic index (adjusting for 1-based numbering)
        selected_topic_data = next(t for t in negative_topics_data if t['topic_num'] == selected_topic_num)
        actual_topic_idx = selected_topic_num - 1
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**Common keywords:** {', '.join(selected_topic_data['words'])}")
            st.write(f"**Issue description:** {st.session_state.negative_topic_descriptions.get(selected_topic_num, 'AI insights loading...')}")
        
        with col2:
            st.metric("Complaints", selected_topic_data['count'], border=True)
        with col3:
            st.metric("Avg Sentiment", f"{selected_topic_data['avg_sentiment']:.2f}", border=True)
        
        # Get sample reviews for this topic
        topic_reviews = negative_df[negative_df['topic'] == actual_topic_idx].copy()
        topic_reviews = topic_reviews.sort_values('sentiment_score')
        
        st.write("**Sample Complaints:**")
        sample_size = min(10, len(topic_reviews))
        for idx, row in topic_reviews.head(sample_size).iterrows():
            with st.container(border=True):
                st.markdown(f"**Severity:** {row['sentiment_score']:.2f} (lower = more severe)")
                st.markdown(f"💬 {row['translated']}")
        
        # Entity Extraction on Negative Reviews
        st.write("**Related Issues & Entities**")
        def extract_entities(text):
            text_lower = str(text).lower()
            entities = []
            feature_keywords = ['login', 'password', 'crash', 'slow', 'freeze', 'bug', 'error', 
                               'loading', 'timeout', 'connection', 'update', 'feature', 'missing',
                               'registration', 'payment', 'transaction', 'security', 'privacy']
            os_keywords = ['android', 'samsung', 'xiaomi', 'oppo', 'vivo', 'tablet', 'phone', 'device']
            
            for keyword in feature_keywords:
                if keyword in text_lower:
                    entities.append((keyword, 'issue'))
            for keyword in os_keywords:
                if keyword in text_lower:
                    entities.append((keyword, 'device'))
            
            seen = set()
            unique_entities = []
            for entity, etype in entities:
                if (entity, etype) not in seen:
                    seen.add((entity, etype))
                    unique_entities.append((entity, etype))
            return unique_entities[:5]
        
        all_entities_neg = []
        for text in topic_reviews['translated']:
            entities = extract_entities(text)
            all_entities_neg.extend(entities)
        
        entity_counts_neg = Counter([entity for entity, _ in all_entities_neg])
        
        if entity_counts_neg:
            entity_data_neg = []
            for entity, count in entity_counts_neg.most_common(10):
                entity_data_neg.append({
                    "Related Issue": entity,
                    "Mentions": count,
                    "Percentage": f"{(count/len(topic_reviews))*100:.0f}%"
                })
            st.dataframe(pd.DataFrame(entity_data_neg), width='content', hide_index=True)
        else:
            st.info("No specific technical issues identified in these complaints")