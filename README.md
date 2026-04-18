# eGovPH Review Insight - Midterm Progress Demo

## 1. Project Introduction

**Project Name**: eGovPH Review Insight
**Web App Name**: eGovPH Review Insight  
**Purpose**: Text Mining and Sentiment Analysis for eGovPH Android App reviews from Google Play Store  
**Tech Stack**: Python, Streamlit, pandas, scikit-learn, NLTK, WordCloud, OpenRouter API

---

## 2. Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Source** | Google Play Store (eGovPH PH Android App) |
| **Total Reviews** | 41,287 |
| **Columns** | `reviewId`, `content`, `score`, `thumbsUpCount`, `reviewCreatedVersion`, `at`, `translated` |
| **Date Range** | April 2026 (data extracted) |
| **App Version** | 2.7.1 (multiple versions supported) |

### Sample Data

| score | reviewCreatedVersion | at | translated |
|-------|---------------------|----|------------|
| 5 | 1.1.2 | 2023-06-02 | sirmaam, do you have someone in the land management bureau that can obtain or take care of the land title because the title has not been taken care of for more than 20 years. |
| 2 | 2.3.5 | 2024-10-06 | always downed cannot use... |
| 4 | 2.6.9 | 2026-01-14 | nice |
| 5 | 2.1.4 | 2024-04-24 | good |
| 4 | 1.8.3 | 2023-10-15 | amazing |
| 5 | 2.3.2 | 2024-08-12 | nice app |
| 5 | 2.6.8 | 2025-07-03 | nice ui. not laggy. would need more updated information. but over all good job. |
| 5 | 2.1.12 | 2024-06-20 | please add back of the id |
| 5 | 2.6.9 | 2026-02-07 | i love this app |
| 5 | 2.6.9 | 2025-11-25 | nice more sent how to get its benefits easier no need to queue |

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

## 7. Workflow Summary

```
Input (CSV) → Streamlit Load → Date Filter → Tab Selection
                                              ↓
                              ┌───────────────┼───────────────┐
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
