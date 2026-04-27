# eGovPH Review Insight - Midterm Progress Demo
[Presentation Link](https://htmlpreview.github.io/?https://github.com/gcbenito1-blip/elective_streamlit/blob/main/presentation.html)

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
| **Total Reviews** | 43,866 |
| **Columns** | `reviewId`, `score`, `thumbsUpCount`, `reviewCreatedVersion`, `at`, `translated` |
| **Date Range** | May 2023 - April 2026 (data extracted) |
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

### Preprocessing Pipeline

The data preprocessing pipeline consists of two main Jupyter notebooks:

#### Step 1: Data Scraping & Cleaning ([`data_preprocessing1.ipynb`](data_preprocessing1.ipynb))

**Tools**: `google-play-scraper`, `langdetect`, `unicodedata`, `re`

**Operations**:
1. **Scrape reviews** from Google Play Store (eGovPH app) using `reviews_all()` with `Sort.NEWEST`
2. **Language detection** using `langdetect` to identify Tagalog (`tl`) vs English reviews
3. **Text cleaning**:
   - Unicode normalization (NFKC)
   - Remove URLs, mentions (@), hashtags (#), HTML tags
   - Remove special symbols and currency symbols
   - Emoji removal using regex pattern
   - Space normalization and lowercasing
4. **Split dataset** into `df_eng` (English) and `df_tgl` (Tagalog)
5. **Export**: `eng_review.csv`, `tgl_review.csv`

#### Step 2: Translation & Dataset Finalization ([`bulk_translate.ipynb`](bulk_translate.ipynb))

**Tools**: `googletrans`, `bulk-translate`, `pandas`

**Operations**:
1. **Batch translate** Tagalog reviews to English using `Translator().translate()` with batch size of 20
2. **Combine** translated Tagalog (`tgl_df['translated']`) with English reviews (`df_eng['content']`)
3. **Sort** by original index to maintain chronological order
4. **Drop** intermediate columns (`clean`, `is_tagalog`)
5. **Parse dates**: Convert `at` column to datetime format (`%Y-%m-%d %H:%M:%S`)
6. **Export**: `final_dataset.csv` (43,907 reviews)

### App-Level Text Processing (Streamlit)

Within the Streamlit app (`elective.py` and `elective_tab/*.py`):
1. **Date filtering**: Sidebar date range picker filters `df` by `at` column
2. **Stopword removal**: NLTK English stopwords + custom app-specific stopwords (`app`, `update`, `version`, `phone`, `po`, `opo`, `ng`, etc.)
3. **Token filtering**: Keep tokens with length > 1
4. **`translated` column**: Used directly for sentiment analysis (VADER) and topic modeling (NMF)

---

## 4. Analytics & Models

### A. Dashboard Overview (eTab1.py)
- **KPIs**: Total Reviews (43,866), Average Score (4.19), Median Score (5.0), Total Likes (26,131)
- **Charts**: Rating Distribution, Reviews per Month, Engagement vs Rating
- **Analysis**: Top App Versions by Volume & Average Score
  - Top versions: 2.6.9 (13,719 reviews), 2.3.5 (6,010), 2.6.8 (1,792), 2.6.2 (1,482), 2.4.3 (1,274)

### B. Text Mining & Sentiment Analysis (eTab2.py)
- **Sentiment Analysis**: VADER-based sentiment scoring and classification (Positive/Neutral/Negative)
- **Sentiment Trends**: Monthly sentiment distribution tracking with percentage breakdowns
- **Topic Modelling**: NMF (Non-negative Matrix Factorization) with 10 topics
- **Word Frequency Analysis**: Top keywords and term prevalence
- **Visualization**: Word Cloud, Sentiment Over Time line charts, Distribution charts
- **Entity Extraction**: Feature and OS/device keyword identification from reviews
- **Co-occurrence Analysis**: Topic-entity relationship mapping
- **AI Enhancement**: OpenRouter API for generating human-readable topic labels

### C. Negative Reviews Deep Dive (eTab3.py)
- **Focus**: Comprehensive analysis of negative sentiment reviews only
- **Method**: VADER sentiment classification (Negative: ≤-0.05)
- **Topic Modeling**: NMF-based topic extraction specific to negative reviews
- **Analysis**: Negative topic distribution, key complaint areas
- **Entity Extraction**: Problem area identification (features, devices, OS issues)
- **AI Enhancement**: OpenRouter API for negative topic labeling and description generation

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
| google-play-scraper | Play Store data extraction |
| langdetect | Language detection (Tagalog/English) |
| googletrans | Batch translation (Tagalog → English) |
| bulk-translate | Async translation helper |
