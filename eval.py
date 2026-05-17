
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def render(df):
    st.header(":material/bar_chart: VADER Evaluation & Validation", anchor=False)
    
    st.markdown("""
    This section evaluates the quality and reliability of VADER sentiment analysis on text analysis.
    Since this is unsupervised learning, we use internal validation metrics to assess sentiment consistency and distribution.
    """)
    
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    
    # Ensure sentiment scores exist
    if 'sentiment_score' not in df.columns:
        def get_sentiment_scores(text):
            if pd.isna(text) or str(text).strip() == '':
                return 0.0
            return sia.polarity_scores(str(text))['compound']
        
        df['sentiment_score'] = df['translated'].apply(get_sentiment_scores)
        df['sentiment'] = df['sentiment_score'].apply(
            lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
        )
    
    # =========================================================================
    # 1. SENTIMENT DISTRIBUTION QUALITY METRICS
    # =========================================================================
    st.subheader("📊 Sentiment Distribution Quality", anchor=False)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_counts = df['sentiment'].value_counts()
        entropy = stats.entropy(sentiment_counts.values / len(df))
        st.metric(
            "Distribution Entropy", 
            f"{entropy:.3f}",
            help="Measures balance between sentiment classes. Higher = more balanced distribution"
        )
    
    with col2:
        # Calculate sentiment polarity balance
        pos_ratio = (df['sentiment'] == 'Positive').mean()
        neg_ratio = (df['sentiment'] == 'Negative').mean()
        balance_score = 1 - abs(pos_ratio - neg_ratio)
        st.metric(
            "Polarity Balance", 
            f"{balance_score:.3f}",
            help="Balance between positive and negative reviews. 1 = perfectly balanced"
        )
    
    with col3:
        # Neutral ratio (should be reasonable, not too high)
        neutral_ratio = (df['sentiment'] == 'Neutral').mean()
        st.metric(
            "Neutral Ratio", 
            f"{neutral_ratio:.1%}",
            help="Percentage of neutral reviews. Very high neutral ratio may indicate issues"
        )
    
    with col4:
        # Confidence score (based on score magnitude)
        avg_magnitude = df['sentiment_score'].abs().mean()
        st.metric(
            "Avg Sentiment Strength", 
            f"{avg_magnitude:.3f}",
            help="Average absolute sentiment score. Higher = more confident classifications"
        )
    
    # Sentiment distribution visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sentiment Class Balance**")
        sentiment_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [
                (df['sentiment'] == 'Positive').sum(),
                (df['sentiment'] == 'Neutral').sum(),
                (df['sentiment'] == 'Negative').sum()
            ]
        })
        
        st.vega_lite_chart(
            sentiment_df,
            {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "width": "container",
                "height": 350,
                "mark": {"type": "arc", "innerRadius": 50},
                "encoding": {
                    "theta": {"field": "Count", "type": "quantitative"},
                    "color": {
                        "field": "Sentiment",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Positive", "Neutral", "Negative"],
                            "range": ["#2ecc71", "#95a5a6", "#e74c3c"]
                        }
                    },
                    "tooltip": [
                        {"field": "Sentiment", "type": "nominal"},
                        {"field": "Count", "type": "quantitative"}
                    ]
                }
            }
        )
    
    with col2:
        st.write("**Sentiment Score Distribution**")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['sentiment_score'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Positive threshold (0.05)')
        ax.axvline(x=-0.05, color='red', linestyle='--', linewidth=2, label='Negative threshold (-0.05)')
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('VADER Compound Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Sentiment Scores', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # =========================================================================
    # 2. CLUSTERING VALIDATION METRICS (Internal Validation)
    # =========================================================================
    st.subheader("🎯 Clustering Validation Metrics", anchor=False)
    st.markdown("These metrics validate if reviews naturally cluster by sentiment without supervision.")
    
    # Prepare text features for clustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['translated'].fillna(''))
    
    # Cluster into 3 groups (positive, neutral, negative)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Silhouette Score (-1 to 1, higher is better)
        sil_score = silhouette_score(tfidf_matrix, cluster_labels, metric='cosine')
        st.metric(
            "Silhouette Score", 
            f"{sil_score:.3f}",
            delta="Good > 0.3" if sil_score > 0.3 else "Poor < 0.3",
            delta_color="normal" if sil_score > 0.3 else "inverse",
            help="Measures how similar reviews are to their own sentiment cluster vs other clusters. Higher = better separation"
        )
    
    with col2:
        # Davies-Bouldin Score (lower is better)
        db_score = davies_bouldin_score(tfidf_matrix.toarray(), cluster_labels)
        st.metric(
            "Davies-Bouldin", 
            f"{db_score:.3f}",
            delta="Good < 1.5" if db_score < 1.5 else "Poor > 1.5",
            delta_color="normal" if db_score < 1.5 else "inverse",
            help="Average similarity between clusters. Lower = more distinct sentiment groups"
        )
    
    with col3:
        # Calinski-Harabasz Score (higher is better)
        ch_score = calinski_harabasz_score(tfidf_matrix.toarray(), cluster_labels)
        st.metric(
            "Calinski-Harabasz", 
            f"{ch_score:.0f}",
            help="Ratio of between-cluster to within-cluster variance. Higher = better defined clusters"
        )
    
    # Compare VADER labels with K-means clusters
    st.write("**VADER vs. Unsupervised Clustering Agreement**")
    
    # Map K-means clusters to sentiment labels
    cluster_sentiment_map = {}
    for cluster in range(3):
        cluster_reviews = df.iloc[cluster_labels == cluster]
        dominant_sentiment = cluster_reviews['sentiment'].mode()[0] if len(cluster_reviews) > 0 else 'Neutral'
        cluster_sentiment_map[cluster] = dominant_sentiment
    
    # Calculate agreement
    predicted_sentiments = [cluster_sentiment_map[label] for label in cluster_labels]
    agreement = (df['sentiment'] == predicted_sentiments).mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "VADER-Clustering Agreement", 
            f"{agreement:.1%}",
            help="Percentage where VADER labels match unsupervised clustering"
        )
        
        # Confusion matrix style comparison
        comparison_df = pd.DataFrame({
            'VADER_Sentiment': df['sentiment'],
            'Cluster_Group': predicted_sentiments
        })
        
        st.write("**Agreement Matrix**")
        agreement_matrix = pd.crosstab(comparison_df['VADER_Sentiment'], comparison_df['Cluster_Group'])
        st.dataframe(agreement_matrix, width='content')
    
    with col2:
        st.write("**Cluster Separation Visualization**")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        sentiment_order = ['Positive', 'Neutral', 'Negative']
        
        for i, sentiment in enumerate(sentiment_order):
            mask = df['sentiment'] == sentiment
            ax.scatter(tfidf_pca[mask, 0], tfidf_pca[mask, 1], 
                      c=colors[i], label=sentiment, alpha=0.6, s=30)
        
        ax.set_xlabel('First Principal Component', fontsize=10)
        ax.set_ylabel('Second Principal Component', fontsize=10)
        ax.set_title('Review Clusters by VADER Sentiment', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # =========================================================================
    # 3. CONSISTENCY ACROSS REVIEW LENGTHS
    # =========================================================================
    st.subheader("📏 Consistency Analysis", anchor=False)
    st.markdown("Evaluating if VADER performs consistently across different review lengths.")
    
    df['review_length'] = df['translated'].str.len()
    df['length_category'] = pd.cut(df['review_length'], 
                                   bins=[0, 50, 200, 500, float('inf')],
                                   labels=['Very Short (<50)', 'Short (50-200)', 'Medium (200-500)', 'Long (500+)'])
    
    # Calculate average sentiment strength by length category
    length_sentiment = df.groupby('length_category').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).round(3)
    length_sentiment.columns = ['Mean Sentiment', 'Std Dev', 'Count']
    
    st.dataframe(length_sentiment, width='content')
    
    # ANOVA test for consistency
    length_groups = [group['sentiment_score'].values for name, group in df.groupby('length_category')]
    if len(length_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*length_groups)
        st.info(f"**ANOVA Test:** F-statistic = {f_stat:.3f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            st.warning("⚠️ Significant differences in sentiment scores across review lengths detected")
        else:
            st.success("✅ Sentiment scores are consistent across review lengths")
    
    # =========================================================================
    # 4. VADER COMPONENT ANALYSIS
    # =========================================================================
    st.subheader("🔍 VADER Component Analysis", anchor=False)
    st.markdown("Breakdown of VADER's four sentiment components.")
    
    # Calculate all VADER components
    pos_scores = []
    neg_scores = []
    neu_scores = []
    compound_scores = []
    
    for text in df['translated'].fillna(''):
        scores = sia.polarity_scores(str(text))
        pos_scores.append(scores['pos'])
        neg_scores.append(scores['neg'])
        neu_scores.append(scores['neu'])
        compound_scores.append(scores['compound'])
    
    df['vader_pos'] = pos_scores
    df['vader_neg'] = neg_scores
    df['vader_neu'] = neu_scores
    
    col1, col2 = st.columns(2)
    
    with col1:
        component_means = pd.DataFrame({
            'Component': ['Positive', 'Negative', 'Neutral'],
            'Average Score': [df['vader_pos'].mean(), df['vader_neg'].mean(), df['vader_neu'].mean()]
        })
        
        st.vega_lite_chart(
            component_means,
            {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "width": "container",
                "height": 350,
                "mark": {"type": "bar", "cornerRadiusEnd": 4},
                "encoding": {
                    "x": {"field": "Component", "type": "nominal", "title": "Sentiment Component"},
                    "y": {"field": "Average Score", "type": "quantitative", "title": "Average Proportion"},
                    "color": {
                        "field": "Component",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Positive", "Neutral", "Negative"],
                            "range": ["#2ecc71", "#95a5a6", "#e74c3c"]
                        }
                    },
                    "tooltip": [
                        {"field": "Component", "type": "nominal"},
                        {"field": "Average Score", "type": "quantitative", "format": ".3f"}
                    ]
                }
            }
        )
    
    with col2:
        st.write("**Component Correlation Matrix**")
        corr_matrix = df[['vader_pos', 'vader_neg', 'vader_neu', 'sentiment_score']].corr()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                    fmt='.3f', square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('VADER Component Correlations', fontsize=12)
        st.pyplot(fig)
    
    # =========================================================================
    # 5. RELIABILITY SCORE & RECOMMENDATIONS
    # =========================================================================
    st.subheader("🎯 VADER Reliability Assessment", anchor=False)
    
    # Calculate overall reliability score (0-100)
    reliability_score = 0
    
    # Factor 1: Distribution balance (20 points)
    entropy_score = min(entropy / 1.5, 1.0) * 20
    reliability_score += entropy_score
    
    # Factor 2: Clustering quality (30 points)
    if sil_score > 0.3:
        clustering_score = min((sil_score - 0.3) / 0.4 * 30, 30)
    else:
        clustering_score = max(sil_score / 0.3 * 30, 0)
    reliability_score += clustering_score
    
    # Factor 3: Sentiment strength (20 points)
    strength_score = min(avg_magnitude / 0.5 * 20, 20)
    reliability_score += strength_score
    
    # Factor 4: Length consistency (15 points)
    if p_value > 0.05 if 'p_value' in locals() else True:
        consistency_score = 15
    else:
        consistency_score = 7.5
    reliability_score += consistency_score
    
    # Factor 5: Low neutral ratio (15 points)
    if neutral_ratio < 0.3:
        neutral_score = 15
    elif neutral_ratio < 0.5:
        neutral_score = 10
    else:
        neutral_score = 5
    reliability_score += neutral_score
    
    reliability_score = min(reliability_score, 100)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(
            "Overall VADER Reliability Score", 
            f"{reliability_score:.0f}/100",
            delta="Good" if reliability_score >= 70 else "Needs Improvement" if reliability_score >= 50 else "Poor",
            delta_color="normal" if reliability_score >= 70 else "off" if reliability_score >= 50 else "inverse"
        )
        
        # Gauge chart
        fig, ax = plt.subplots(figsize=(6, 2))
        colors_gauge = ['#e74c3c', '#f39c12', '#2ecc71']
        
        # Create horizontal bar gauge
        ax.barh([0], [100], left=0, color='#ecf0f1', height=0.3)
        if reliability_score < 50:
            color = '#e74c3c'
        elif reliability_score < 70:
            color = '#f39c12'
        else:
            color = '#2ecc71'
        ax.barh([0], [reliability_score], left=0, color=color, height=0.3)
        
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0', '25', '50', '75', '100'])
        ax.set_title(f'Reliability Score: {reliability_score:.0f}/100', fontsize=12, pad=10)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=70, color='gray', linestyle='--', alpha=0.5)
        st.pyplot(fig)
    
    with col2:
        st.write("**Component Scores**")
        component_scores = pd.DataFrame({
            'Metric': ['Distribution Balance', 'Clustering Quality', 'Sentiment Strength', 'Length Consistency', 'Low Neutral Ratio'],
            'Score': [f"{entropy_score:.0f}/20", f"{clustering_score:.0f}/30", f"{strength_score:.0f}/20", f"{consistency_score:.0f}/15", f"{neutral_score:.0f}/15"]
        })
        st.dataframe(component_scores, width='content', hide_index=True)
    
    # Recommendations
    st.write("**Recommendations**")
    
    recommendations = []
    if neutral_ratio > 0.5:
        recommendations.append("• **High neutral ratio detected** - Consider using aspect-based sentiment analysis or reviewing ambiguous cases")
    if sil_score < 0.3:
        recommendations.append("• **Poor clustering separation** - Reviews may need better preprocessing or additional context")
    if avg_magnitude < 0.3:
        recommendations.append("• **Low sentiment strength** - Reviews may be too neutral or brief for clear classification")
    if p_value < 0.05 if 'p_value' in locals() else False:
        recommendations.append("• **Length bias detected** - VADER performs differently on short vs long reviews")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ VADER is performing well on your review data! The sentiment analysis appears reliable.")
    
    # =========================================================================
    # 6. ERROR ANALYSIS (Ambiguous Cases)
    # =========================================================================
    st.subheader("⚠️ Ambiguous Cases Analysis", anchor=False)
    st.markdown("Reviews where VADER shows uncertainty or potential misclassification.")
    
    # Identify uncertain cases
    df['uncertainty'] = 1 - (df['vader_pos'] + df['vader_neg'])  # High uncertainty when both pos and neg are low
    df['sentiment_strength_abs'] = df['sentiment_score'].abs()
    
    ambiguous_df = df[
        (df['sentiment_strength_abs'] < 0.2) |  # Weak sentiment
        (df['uncertainty'] > 0.7)  # High uncertainty
    ].head(10)
    
    if len(ambiguous_df) > 0:
        st.write("**Potentially Misclassified or Unclear Reviews:**")
        for idx, row in ambiguous_df.iterrows():
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"💬 {row['translated'][:150]}...")
                with col2:
                    st.write(f"**Sentiment:** {row['sentiment']}")
                with col3:
                    st.write(f"**Score:** {row['sentiment_score']:.3f}")
    else:
        st.info("No highly ambiguous cases detected. All reviews have clear sentiment signals.")
df = pd.read_csv('data/egovph.csv')
render(df)