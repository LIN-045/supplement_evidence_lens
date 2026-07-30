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
    "nih_ods_guidance": "NIH ODS Consumer Guidance",
    "us_dri_tables": "Dietary Reference Intakes",
    "nccih_herbs": "NCCIH Herbs at a Glance",
    "us_nih_ods_faq": "NIH ODS Consumer FAQ",
}


st.set_page_config(
    page_title="Monitoring · Supplement Evidence Lens",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url(
            "https://fonts.googleapis.com/css2?family=Expletus+Sans:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap"
        );

        .stApp {
            background:
                radial-gradient(circle at 90% 0%, #e4f2ed 0, transparent 30rem),
                #f7faf8;
            color: #29433b;
        }

        .stApp,
        .stApp button,
        .stApp input,
        .stApp textarea {
            font-family:
                "Inter", "Avenir Next", "Segoe UI", Helvetica, Arial,
                sans-serif !important;
            font-weight: 400;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 5.25rem;
            padding-bottom: 5.25rem;
        }

        .monitoring-lockup {
            align-items: flex-end;
            display: flex;
            gap: 0.9rem;
        }

        .monitoring-mark {
            flex: 0 0 auto;
            height: 3rem;
            transform: translateY(-1.15rem);
            width: 3rem;
        }

        .monitoring-kicker {
            color: #65746f;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.13em;
            margin: 0 0 0.35rem;
            text-transform: uppercase;
        }

        .monitoring-title {
            color: #234c42 !important;
            font-family:
                "Expletus Sans", "Inter", "Avenir Next",
                sans-serif !important;
            font-size: clamp(1.9rem, 4vw, 2.65rem);
            font-weight: 500 !important;
            letter-spacing: -0.02em;
            line-height: 1.05;
            margin: 0;
        }

        .monitoring-copy {
            color: #65746f;
            font-size: 0.98rem;
            line-height: 1.65;
            margin: 0.75rem 0 2.25rem 3.9rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce8e3;
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #65746f;
        }

        div[data-testid="stMetricValue"] {
            color: #245e4c;
        }

        .st-key-questions_card,
        .st-key-latency_card,
        .st-key-feedback_card,
        .st-key-searches_card,
        .st-key-sources_card,
        .st-key-interactions_card {
            background: #ffffff;
            border-color: #dce8e3 !important;
            border-radius: 14px;
            margin-top: 1rem;
            padding: 0.35rem 1rem 1rem;
        }

        .st-key-questions_card {
            border-top: 3px solid #245e4c !important;
        }

        .st-key-latency_card {
            border-top: 3px solid #3f7564 !important;
        }

        .st-key-feedback_card {
            border-top: 3px solid #5b8c7b !important;
        }

        .st-key-searches_card {
            border-top: 3px solid #72a091 !important;
        }

        .st-key-sources_card {
            border-top: 3px solid #8ab5a6 !important;
        }

        .st-key-interactions_card {
            border-top: 3px solid #b9ddd0 !important;
        }

        h3 {
            color: #294b41;
            font-size: 1.08rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.015em;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 3.5rem;
                padding-bottom: 3.5rem;
            }

            .monitoring-title {
                font-size: clamp(1.7rem, 8vw, 2.2rem);
            }

            .monitoring-copy {
                margin-left: 0;
            }
        }
    </style>
    <div class="monitoring-lockup">
        <svg
            class="monitoring-mark"
            viewBox="0 0 48 48"
            role="img"
            aria-label="Supplement Evidence Lens monitoring logo"
        >
            <rect
                x="1"
                y="1"
                width="46"
                height="46"
                rx="13"
                fill="#245E4C"
            />
            <path
                d="M10 31L17 24L23 28L31 18L38 22"
                fill="none"
                stroke="#FFFFFF"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2.5"
            />
            <circle cx="17" cy="24" r="2.25" fill="#B9DDD0"/>
            <circle cx="31" cy="18" r="2.25" fill="#B9DDD0"/>
        </svg>
        <div>
            <p class="monitoring-kicker">System health and quality</p>
            <h1 class="monitoring-title">
                Supplement Evidence Lens Monitoring
            </h1>
        </div>
    </div>
    <p class="monitoring-copy">
        Usage, response performance, retrieval behavior, citations, and
        user feedback.
    </p>
    """,
    unsafe_allow_html=True,
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
    with st.container(border=True, key="questions_card"):
        st.subheader("Questions Over Time")
        st.caption("Daily application usage")
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
            color="#245e4c",
        )

with right_column:
    with st.container(border=True, key="latency_card"):
        st.subheader("Average Response Time")
        st.caption("Daily end-to-end latency in seconds")
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
            color="#3f7564",
        )

with left_column:
    with st.container(border=True, key="feedback_card"):
        st.subheader("User Feedback")
        st.caption("Helpful, unhelpful, and unrated answers")
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
            color="#5b8c7b",
        )

with right_column:
    with st.container(border=True, key="searches_card"):
        st.subheader("Searches Per Question")
        st.caption("Agent retrieval depth")
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
            color="#72a091",
        )

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

with st.container(border=True, key="sources_card"):
    st.subheader("Cited Sources")
    st.caption("Evidence usage across official collections")
    st.bar_chart(
        source_chart,
        x="source",
        y="citations",
        horizontal=True,
        color="#245e4c",
    )

with st.container(border=True, key="interactions_card"):
    st.subheader("Recent Interactions")
    st.caption("Latest questions and application behavior")
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
            "context_count": "Cited Contexts",
            "response_time_seconds": st.column_config.NumberColumn(
                "Response Time",
                format="%.1f s",
            ),
            "feedback": "Feedback",
        },
        hide_index=True,
        width="stretch",
    )
