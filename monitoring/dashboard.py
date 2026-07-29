"""Streamlit monitoring dashboard for Supplement Evidence Lens."""

from collections import Counter
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.store import load_interactions


SOURCE_LABELS = {
    "eu_health_claims_register": "EU Register",
    "health_canada_nhpid": "Health Canada",
    "nih_ods": "NIH ODS",
}


st.set_page_config(
    page_title="Monitoring · Supplement Evidence Lens",
    page_icon="◉",
    layout="wide",
)

st.title("Application Monitoring")
st.caption(
    "Usage, performance, retrieval behavior, citations, and user feedback."
)

records = load_interactions()

if not records:
    st.info(
        "No interactions have been recorded yet. "
        "Ask a question in the main application first."
    )
    st.stop()

dataframe = pd.DataFrame(records)
dataframe["created_at"] = pd.to_datetime(
    dataframe["created_at"],
    utc=True,
)
dataframe["date"] = dataframe["created_at"].dt.date

feedback_count = int(dataframe["feedback"].notna().sum())
positive_count = int((dataframe["feedback"] == 1).sum())
positive_rate = (
    positive_count / feedback_count
    if feedback_count
    else 0.0
)

metric_columns = st.columns(4)
metric_columns[0].metric("Questions", len(dataframe))
metric_columns[1].metric(
    "Average Response Time",
    f"{dataframe['response_time_seconds'].mean():.1f}s",
)
metric_columns[2].metric(
    "Average Searches",
    f"{dataframe['search_count'].mean():.2f}",
)
metric_columns[3].metric(
    "Positive Feedback",
    f"{positive_rate:.0%}" if feedback_count else "No ratings",
)

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("1. Questions Over Time")
    daily_questions = (
        dataframe.groupby("date")
        .size()
        .rename("questions")
        .reset_index()
    )
    st.line_chart(
        daily_questions,
        x="date",
        y="questions",
    )

with right_column:
    st.subheader("2. Average Response Time")
    daily_latency = (
        dataframe.groupby("date")["response_time_seconds"]
        .mean()
        .rename("seconds")
        .reset_index()
    )
    st.line_chart(
        daily_latency,
        x="date",
        y="seconds",
    )

with left_column:
    st.subheader("3. User Feedback")
    feedback_labels = dataframe["feedback"].map(
        {
            1: "Positive",
            -1: "Negative",
        }
    ).fillna("No feedback")
    feedback_chart = (
        feedback_labels.value_counts()
        .rename_axis("rating")
        .rename("responses")
        .reset_index()
    )
    st.bar_chart(
        feedback_chart,
        x="rating",
        y="responses",
    )

with right_column:
    st.subheader("4. Searches Per Question")
    search_chart = (
        dataframe["search_count"]
        .value_counts()
        .sort_index()
        .rename_axis("searches")
        .rename("questions")
        .reset_index()
    )
    st.bar_chart(
        search_chart,
        x="searches",
        y="questions",
    )

st.subheader("5. Cited Sources")
source_counts: Counter[str] = Counter()

for contexts in dataframe["contexts"]:
    for context in contexts:
        source_counts[
            SOURCE_LABELS.get(
                context["source"],
                context["source"],
            )
        ] += 1

source_chart = pd.DataFrame(
    [
        {
            "source": source,
            "citations": count,
        }
        for source, count in source_counts.most_common()
    ]
)
st.bar_chart(
    source_chart,
    x="source",
    y="citations",
    horizontal=True,
)

st.subheader("Recent Interactions")
recent_interactions = dataframe.sort_values(
    "created_at",
    ascending=False,
).head(10)
recent_interactions = recent_interactions.assign(
    feedback=recent_interactions["feedback"].map(
        {
            1: "Positive",
            -1: "Negative",
        }
    ).fillna("—")
)

st.dataframe(
    recent_interactions[
        [
            "created_at",
            "question",
            "search_count",
            "context_count",
            "response_time_seconds",
            "feedback",
        ]
    ],
    column_config={
        "created_at": "Time",
        "question": "Question",
        "search_count": "Searches",
        "context_count": "Sources",
        "response_time_seconds": st.column_config.NumberColumn(
            "Response Time",
            format="%.1f s",
        ),
        "feedback": "Feedback",
    },
    hide_index=True,
    use_container_width=True,
)
