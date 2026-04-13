# eGovPH Review Insight - Midterm Progress Demo

## 1. Project Introduction

**Project Name**: eGovPH Review Insight  
**Purpose**: Text Mining and Sentiment Analysis for eGovPH Android App reviews from Google Play Store  
**Tech Stack**: Python, Streamlit, pandas, scikit-learn, NLTK, WordCloud, OpenRouter API

---

## 2. Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Source** | Google Play Store (eGovPH PH Android App) |
| **Total Reviews** | 41,287 |
| **Final dataset Columns** | `reviewId`, `content`, `score`, `thumbsUpCount`, `reviewCreatedVersion`, `at`, `translated` |
| **Date Range** | Feb 2026 (data extracted) |
| **App Version** | 2.6.9 |

### Sample Data

| score | reviewCreatedVersion | at | translated |
|-------|---------------------|----|------------|
| 5 | 2.6.9 | 2026-02-17 | good |
| 5 | 2.6.9 | 2026-02-16 | exelent |
| 5 | 2.6.9 | 2026-02-16 | happy |

---

## 3. Data Processing / Preparation

### Preprocessing Steps Implemented:

1. **Date Parsing**: Converted `at` column to datetime format
2. **Text Cleaning**: 
   - Removed English stopwords (NLTK)
   - Removed app-specific stopwords (e.g., "app", "update", "version", "phone", "po", "opo", "ng")
   - Filtered tokens with length > 1
3. **Filtering**: Date range filter via sidebar (start/end date picker)

---

## 4. Analytics & Models

### A. Dashboard Overview (eTab1.py)
- **KPIs**: Total Reviews, Average Score, Median Score, Total Likes
- **Charts**: Rating Distribution, Reviews per Day, Engagement vs Rating
- **Analysis**: Top App Versions by Volume & Average Score

### B. Text Mining Analysis (eTab2.py)
- **Topic Modelling**: NMF (Non-negative Matrix Factorization) with 10 topics
- **Word Frequency**: Top 20 most common words
- **Visualization**: Word Cloud
- **AI Enhancement**: OpenRouter API for generating human-readable topic labels

### C. Sentiment Analysis (eTab3.py)
- **Method**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Classification**: Positive (≥0.05), Neutral (-0.05 to 0.05), Negative (≤-0.05)
- **Analysis**: Sentiment distribution, score histogram, negative review topics
- **AI Enhancement**: OpenRouter API for negative review topic labeling

---

## 5. System/Prototype Demo

### Running the Application
```bash
streamlit run elective.py
```

### Features Demonstrated:
| Tab | Feature | Description |
|-----|---------|-------------|
| Overview | Dashboard | KPIs, charts, review listing |
| Text Mining | Topic Modelling | 10 topics extracted, word cloud |
| Sentiment | VADER Analysis | 3-class sentiment classification |

### Key Visualizations:
- Rating distribution bar chart
- Reviews per day line chart
- Sentiment distribution pie/bar chart
- Word cloud generation

---

## 6. Current Limitations & Next Steps

### Limitations:
- [ ] Dataset limited to single time period (Feb 2026)
- [ ] No machine learning model for rating prediction
- [ ] AI topic labeling requires API key
- [ ] No export functionality for analyzed data
- [ ] Limited language support (English only after translation)

### Next Steps (To Complete):
1. **Rating Prediction Model**: Build ML model to predict review score based on text
2. **Enhanced Visualizations**: Add interactive charts with drill-down
3. **Data Export**: Allow CSV export of analyzed results
4. **Comparison Analysis**: Compare sentiment across app versions
5. **Time Series Analysis**: Trend analysis over time

---

## 7. Workflow Summary

```
Input (CSV) → Streamlit Load → Date Filter → Tab Selection
                                               ↓
                              ┌──────────���────┼───────────────┐
                              ↓               ↓               ↓
                         eTab1.py         eTab2.py         eTab3.py
                              ↓               ↓               ↓
                        Dashboard       Text Mining     Sentiment
                              ↓               ↓               ↓
                        Metrics/Charts  Topics/WordCloud  VADER Scores
```

---

## 8. Tools & Libraries Used

| Library | Purpose |
|---------|---------|
| Streamlit | Web UI Framework |
| pandas | Data Processing |
| scikit-learn | Topic Modelling (NMF, TF-IDF) |
| NLTK | Stopwords, VADER Sentiment |
| WordCloud | Word Cloud Visualization |
| matplotlib | Plotting |
| OpenRouter API | AI Topic Labeling |
| requests | HTTP Requests |
