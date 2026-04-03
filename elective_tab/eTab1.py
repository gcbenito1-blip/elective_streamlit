import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd


def render(df):
    st.header(":material/dashboard: Dataset Overview", anchor=False)
    df = pd.read_csv('data/egovph.csv')
    df['at'] = pd.to_datetime(df['at'])
    df = df.sort_values('at')
    st.markdown("**Key Performance Indicators**")
    col1, col2, col3, col4 = st.columns(4, border=True)
    col1.metric("Total Reviews", len(df))
    col2.metric("Average Score", round(df['score'].mean(), 2))
    col3.metric("Median Score", df['score'].median())
    col4.metric("Total Likes", int(df['thumbsUpCount'].sum()))
    score_counts = df['score'].value_counts().sort_index()
    chart_df = score_counts.reset_index()
    chart_df.columns = ['score', 'count']
    spec = {
        "title":"App Rating Distribution",
        "data": {"values": chart_df.to_dict(orient="records")},
        "mark": "bar",
        "encoding": {
            "y": {
                "field": "score",
                "type": "ordinal",
                "sort": "descending",
                "title": "Score"
            },
            "x": {
                "field": "count",
                "type": "quantitative",
                "title": "Count"
            }
        }
    }

    colA, colB = st.columns(2)
    with colA:
        with st.container():
            st.vega_lite_chart(spec,width="stretch")
    
    eng = df.groupby('score')['thumbsUpCount'].mean().reset_index()
    eng['thumbsUpCount'] = eng['thumbsUpCount'].round(4)

    with colB:
        with st.container():
            st.markdown("**Engagement vs Rating**")
            st.bar_chart(eng.set_index('score'), x_label="Score", y_label="Average Review Likes", horizontal=True)
    
    top_versions = df['reviewCreatedVersion'].value_counts()
    ver_df = top_versions.reset_index()
    ver_df.columns = ['version', 'count']

    st.subheader("Top Versions by Volume")
    st.bar_chart(ver_df.set_index('version'))

    avg_ver = df.groupby('reviewCreatedVersion')['score'].mean().sort_values(ascending=False)
    st.subheader("Top Versions by Avg Score")
    st.bar_chart(avg_ver)

    col_1, col_2 = st.columns(2, border=True)
    ## Recent Low Scores
    with col_1:
        low = df[df['score'] <= 2].sort_values(by='at', ascending=False)[
                ['translated', 'score', 'thumbsUpCount']
            ].head(10)

        st.subheader("Recent Low Scores", anchor=False)

        for _, row in low.iterrows():
            st.markdown(f"""
        **Score:** {row['score']} :material/star:| **Likes:** {row['thumbsUpCount']} :material/thumb_up:

        {row['translated']}

        ---
        """)

    ## Most Upvoted Reviews
    with col_2:
        top = df.sort_values(by='thumbsUpCount', ascending=False).head(10)
        st.subheader("Most Upvoted Reviews",anchor=False)
        for _, row in top.iterrows():
            st.markdown(f"""
        **Score:** {row['score']}:material/star: | **Likes:** {row['thumbsUpCount']} :material/thumb_up:

        {row['translated']}

        ---
        """)