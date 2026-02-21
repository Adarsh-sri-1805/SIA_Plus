import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymongo
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import time

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="SIA+ Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
defaults = {
    "chat_active": False,
    "chat_pending": False,
    "chat_answer": "",
    "summary_text": "",
    "summary_generated": False,
    "last_refresh": datetime.now()
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# CONFIGURATION
# -----------------------------
MONGO_URI = "mongodb+srv://adarsh2428cseai1180_db_user:D3m7CqvVMupyw1nl@sia.4jl1u4u.mongodb.net/?appName=SIA"
DB_NAME = "sia_plus"
COLLECTION_NAME = "reviews"
API_BASE_URL = "http://localhost:8000"

# -----------------------------
# AUTO REFRESH LOGIC (intelligent)
# -----------------------------
def should_auto_refresh():
    return not (st.session_state.chat_active or st.session_state.chat_pending)

if should_auto_refresh():
    st_autorefresh(interval=30000, key="auto-refresh")

# -----------------------------
# DATABASE CONNECTION (cached)
# -----------------------------
@st.cache_resource(ttl=300)
def get_collection():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

collection = get_collection()

@st.cache_data(ttl=10)  # cache data for 10 seconds
def fetch_reviews():
    return list(collection.find({}, {"_id": 0}))

# -----------------------------
# LOAD DATA
# -----------------------------
reviews = fetch_reviews()

if not reviews:
    st.warning("No reviews found. Please check your database connection.")
    st.stop()

df = pd.DataFrame(reviews)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()

# -----------------------------
# AUTO-GENERATE EXECUTIVE SUMMARY (once per session)
# -----------------------------
if not st.session_state.summary_generated:
    with st.spinner("Generating executive summary..."):
        try:
            res = requests.get(f"{API_BASE_URL}/summary", timeout=120).json()
            st.session_state.summary_text = res.get("summary", "No summary available.")
        except Exception as e:
            st.session_state.summary_text = f"Error generating summary: {e}"
        st.session_state.summary_generated = True
        st.rerun()

# -----------------------------
# SIDEBAR: FILTERS & CONTROLS
# -----------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=SIA+LOGO", width=True)  # replace with your logo
    st.title("🔧 Controls")
    st.markdown("---")
    
    # Date range filter
    st.subheader(" Date Range")
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    start_date = st.date_input("Start", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End", max_date, min_value=min_date, max_value=max_date)
    
    # Sentiment filter
    st.subheader("Sentiment")
    sentiment_options = ["Positive", "Negative", "Neutral"]
    selected_sentiments = st.multiselect("Select", sentiment_options, default=sentiment_options)
    
    # Category filter (if exists)
    if "category" in df.columns:
        st.subheader("Categories")
        categories = df["category"].dropna().unique().tolist()
        selected_categories = st.multiselect("Select categories", categories, default=[])
    else:
        selected_categories = []
    
    st.markdown("---")
    st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# -----------------------------
# APPLY FILTERS
# -----------------------------
mask = (
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date) &
    (df["sentiment"].isin(selected_sentiments))
)
if selected_categories:
    mask &= (df["category"].isin(selected_categories))
filtered_df = df[mask].copy()

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# -----------------------------
# MAIN DASHBOARD HEADER
# -----------------------------
st.title("SIA+ Real-Time Customer Intelligence")
st.markdown(f"**{len(filtered_df)} reviews** from **{start_date}** to **{end_date}**")

# -----------------------------
# EXECUTIVE SUMMARY CARD
# -----------------------------
with st.container():
    st.subheader("Executive Summary")
    summary_col1, summary_col2 = st.columns([5,1])
    with summary_col1:
        st.info(st.session_state.summary_text)
    with summary_col2:
        if st.button("⟲ Regenerate"):
            st.session_state.summary_generated = False
            st.rerun()

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total = len(filtered_df)
positive = (filtered_df["sentiment"] == "Positive").sum()
negative = (filtered_df["sentiment"] == "Negative").sum()
neutral = (filtered_df["sentiment"] == "Neutral").sum()

with kpi1:
    st.metric("Total Reviews", total)
with kpi2:
    st.metric("Positive", positive, f"{positive/total*100:.1f}%" if total else "0%")
with kpi3:
    st.metric("Negative", negative, f"{negative/total*100:.1f}%" if total else "0%")
with kpi4:
    st.metric("Neutral", neutral, f"{neutral/total*100:.1f}%" if total else "0%")
with kpi5:
    # Average sentiment score (if we had numeric, but we can create a simple score)
    # For now, just show a ratio of positive/negative
    if negative > 0:
        ratio = positive / negative
    else:
        ratio = positive
    st.metric("Pos/Neg Ratio", f"{ratio:.2f}")

# -----------------------------
# ALERT SYSTEM
# -----------------------------
one_min_ago = datetime.now() - timedelta(minutes=1)
recent_neg = filtered_df[
    (filtered_df["sentiment"] == "Negative") &
    (filtered_df["timestamp"] >= one_min_ago)
]
if len(recent_neg) >= 3:
    st.error("**ALERT:** 3+ negative reviews in the last minute – investigate immediately!")

# -----------------------------
# ROW 1: DISTRIBUTION CHARTS
# -----------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader(" Sentiment Distribution")
    fig_pie = px.pie(
        filtered_df,
        names="sentiment",
        title="Overall Sentiment",
        hole=0.4,
        color="sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with row1_col2:
    st.subheader(" Sentiment Counts")
    counts = filtered_df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig_bar = px.bar(
        counts,
        x="sentiment",
        y="count",
        color="sentiment",
        title="Number of Reviews by Sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# ROW 2: TIME SERIES CHARTS
# -----------------------------
st.subheader(" Sentiment Trends")

# Daily stacked area
daily = filtered_df.groupby(["date", "sentiment"]).size().reset_index(name="count")
fig_area = px.area(
    daily,
    x="date",
    y="count",
    color="sentiment",
    title="Daily Review Volume by Sentiment",
    color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"},
    line_shape="linear"
)
fig_area.update_layout(xaxis_title="Date", yaxis_title="Number of Reviews")
st.plotly_chart(fig_area, use_container_width=True)

# Hourly trend (if multiple days)
if len(filtered_df["date"].unique()) > 1:
    hourly = filtered_df.groupby(["hour", "sentiment"]).size().reset_index(name="count")
    fig_hour = px.line(
        hourly,
        x="hour",
        y="count",
        color="sentiment",
        title="Hourly Review Pattern (All Days Combined)",
        markers=True,
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
    )
    fig_hour.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
    st.plotly_chart(fig_hour, use_container_width=True)

# Day of week analysis
if len(filtered_df) > 10:
    dow = filtered_df.groupby(["day_of_week", "sentiment"]).size().reset_index(name="count")
    # Order days correctly
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=day_order, ordered=True)
    dow = dow.sort_values("day_of_week")
    fig_dow = px.bar(
        dow,
        x="day_of_week",
        y="count",
        color="sentiment",
        title="Reviews by Day of Week",
        barmode="group",
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
    )
    st.plotly_chart(fig_dow, use_container_width=True)

# -----------------------------
# ROW 3: CATEGORY ANALYSIS (if available)
# -----------------------------
if "category" in filtered_df.columns and not filtered_df["category"].isna().all():
    st.subheader(" Category Analysis")
    col_cat1, col_cat2 = st.columns(2)
    
    with col_cat1:
        cat_counts = filtered_df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig_cat = px.bar(
            cat_counts,
            x="category",
            y="count",
            color="category",
            title="Reviews by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_cat2:
        cat_sent = filtered_df.groupby(["category", "sentiment"]).size().reset_index(name="count")
        fig_cat_sent = px.bar(
            cat_sent,
            x="category",
            y="count",
            color="sentiment",
            title="Sentiment by Category",
            barmode="group",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
        )
        st.plotly_chart(fig_cat_sent, use_container_width=True)

# -----------------------------
# RECENT NEGATIVE REVIEWS TABLE
# -----------------------------
st.subheader(" Recent Negative Reviews")
neg_df = filtered_df[filtered_df["sentiment"] == "Negative"].sort_values(by="timestamp", ascending=False).head(5)
if not neg_df.empty:
    display_cols = ["text", "timestamp"]
    if "category" in neg_df.columns:
        display_cols.insert(1, "category")
    st.dataframe(
        neg_df[display_cols].style.applymap(lambda x: 'color: #e74c3c', subset=['text']),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No negative reviews in selected period.")

# -----------------------------
# BOTTOM SECTION: SEARCH & CHAT
# -----------------------------
st.markdown("---")
st.subheader(" Semantic Search & AI Assistant")

search_col, chat_col = st.columns(2)

with search_col:
    st.markdown("**Find similar reviews**")
    search_query = st.text_input("Enter keywords or phrase", key="search_input")
    if search_query:
        with st.spinner("Searching..."):
            try:
                res = requests.get(f"{API_BASE_URL}/search", params={"query": search_query}, timeout=30).json()
                results = res.get("results", [])
                if results:
                    for i, r in enumerate(results[:5], 1):
                        st.markdown(f"**{i}.** {r}")
                else:
                    st.info("No similar reviews found.")
            except Exception as e:
                st.error(f"Search error: {e}")

with chat_col:
    st.markdown("**Ask AI about reviews**")
    question = st.text_input("Your question", key="chat_input")
    if st.button("Analyze", type="primary"):
        st.session_state.chat_active = True
        st.session_state.chat_pending = True
        with st.spinner("Generating answer..."):
            try:
                res = requests.get(f"{API_BASE_URL}/chat", params={"query": question}, timeout=60).json()
                st.session_state.chat_answer = res.get("answer", "No answer returned.")
            except Exception as e:
                st.session_state.chat_answer = f"Chat error: {e}"
            finally:
                st.session_state.chat_pending = False
        st.rerun()
    
    if st.session_state.chat_answer:
        st.info(st.session_state.chat_answer)
    
    if st.button("Clear Chat"):
        st.session_state.chat_answer = ""
        st.session_state.chat_active = False
        st.session_state.chat_pending = False
        st.rerun()

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(" Powered by SIA+ AI | Auto-refresh every 30 seconds (paused during chat)")