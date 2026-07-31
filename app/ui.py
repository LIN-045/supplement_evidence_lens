"""Streamlit interface for Supplement Evidence Lens.

Run from the project root:

    uv run streamlit run app/ui.py
"""

from pathlib import Path
import sys
from time import perf_counter
from typing import Any
from collections import defaultdict
from html import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from app.agentic_rag import AgenticRAG
from app.retrieval import EvidenceRetriever
from monitoring.store import record_feedback, record_interaction


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
    page_title="Supplement Evidence Lens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
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
            max-width: 1040px;
            padding-top: 4.75rem;
            padding-bottom: 4rem;
        }

        .brand-lockup {
            align-items: center;
            display: flex;
            gap: 0.8rem;
            justify-content: center;
        }

        .brand-mark {
            flex: 0 0 auto;
            height: 2.7rem;
            width: 2.7rem;
        }

        h1.hero-title {
            color: #234c42 !important;
            font-family:
                "Expletus Sans", "Inter", "Avenir Next",
                sans-serif !important;
            font-size: clamp(1.9rem, 4vw, 2.65rem);
            font-weight: 500 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.05;
            margin: 0;
            white-space: nowrap;
        }

        .hero-copy {
            color: #65746f;
            font-size: 0.98rem;
            line-height: 1.65;
            margin: 0.65rem auto 1.7rem;
            text-align: center;
        }

        .disclaimer {
            border-top: 1px solid #dce6e1;
            color: #687a74;
            font-size: 0.8rem;
            line-height: 1.5;
            margin-top: 3rem;
            padding-top: 1rem;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #dce8e3;
            border-radius: 12px;
            box-shadow: 0 3px 12px rgba(35, 39, 70, 0.04);
            padding: 0.9rem 1rem 0.65rem;
        }

        div[data-testid="stTextArea"] label p {
            color: #29433b;
            font-size: 1rem;
            font-weight: 600;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: #245e4c;
            border-color: #245e4c;
            border-radius: 8px;
            color: #ffffff;
            min-height: 2.75rem;
        }

        div[data-testid="stFormSubmitButton"] > button p {
            font-size: 0.96rem;
            font-weight: 600;
        }

        div[data-testid="stFormSubmitButton"] > button {
            min-height: 2.4rem;
            padding: 0.45rem 1rem;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #194b3d;
            border-color: #194b3d;
            color: #ffffff;
        }

        .st-key-feedback_panel {
            margin-top: 1.2rem;
        }

        .st-key-feedback_panel div[data-testid="stCaptionContainer"] {
            margin-bottom: -0.35rem;
        }

        .st-key-feedback_panel div[data-testid="stFeedback"] {
            opacity: 0.72;
        }

        .st-key-feedback_panel div[data-testid="stFeedback"] button + button {
            margin-left: 0.55rem;
        }

        .st-key-feedback_panel div[data-testid="stPopover"] button {
            color: #65746f;
            height: 2.5rem;
            min-height: 2.5rem;
            min-width: 2.5rem;
            padding: 0.35rem;
        }

        .st-key-feedback_panel div[data-testid="stPopover"] button p {
            display: none;
        }

        .st-key-feedback_panel
        div[data-testid="stPopover"]
        span[data-testid="stIconMaterial"] {
            font-size: 1.5rem;
        }

        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #dce8e3;
            border-radius: 8px;
        }

        div[data-testid="stExpander"] summary p {
            color: #294b41;
            font-weight: 500;
        }

        .evidence-item {
            border-bottom: 1px solid #e6eeea;
            margin-bottom: 0.9rem;
            padding-bottom: 0.9rem;
        }

        .evidence-item:last-child {
            border-bottom: 0;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        .evidence-title {
            font-size: 0.98rem;
            font-weight: 500;
            line-height: 1.4;
        }

        .evidence-title a {
            color: #245e4c !important;
            text-decoration: none !important;
        }

        .evidence-title a:hover {
            text-decoration: underline !important;
        }

        .evidence-title-label {
            color: #245e4c;
        }

        h2 {
            color: #18352d;
            font-size: 1.75rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em;
            margin-top: 1.8rem !important;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 2.75rem;
            }

            h1.hero-title {
                font-size: clamp(1.7rem, 8vw, 2.2rem);
                text-align: center;
                white-space: normal;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading search models...")
def load_rag() -> AgenticRAG:
    """Create the long-lived RAG service once per Streamlit process."""

    load_dotenv(PROJECT_ROOT / ".env")
    return AgenticRAG(
        EvidenceRetriever.from_defaults(),
        OpenAI(),
    )


def show_contexts(contexts: list[dict[str, Any]]) -> None:
    """Group cited excerpts by source in a compact evidence list."""

    st.header("Sources")

    if not contexts:
        st.info("The answer did not cite a retrieved source.")
        return

    grouped_contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        grouped_contexts[context["source"]].append(context)

    for source_id, source_contexts in grouped_contexts.items():
        source = SOURCE_LABELS.get(
            source_id,
            source_id,
        )
        reference_numbers = ", ".join(
            f"[{context['reference']}]"
            for context in source_contexts
        )
        excerpt_label = (
            "excerpt"
            if len(source_contexts) == 1
            else "excerpts"
        )

        with st.expander(
            f"{source} · {len(source_contexts)} cited {excerpt_label} "
            f"{reference_numbers}"
        ):
            contexts_by_title: dict[
                str,
                list[dict[str, Any]],
            ] = defaultdict(list)
            for context in source_contexts:
                contexts_by_title[str(context["title"])].append(
                    context
                )

            for raw_title, title_contexts in contexts_by_title.items():
                title = escape(raw_title)

                if len(title_contexts) == 1:
                    context = title_contexts[0]
                    source_url = escape(
                        str(context["source_url"]),
                        quote=True,
                    )
                    source_links = (
                        f'<a href="{source_url}" target="_blank">'
                        f'[{context["reference"]}] {title} ↗</a>'
                    )
                else:
                    reference_links = []
                    for context in title_contexts:
                        source_url = escape(
                            str(context["source_url"]),
                            quote=True,
                        )
                        reference_links.append(
                            f'<a href="{source_url}" target="_blank">'
                            f'[{context["reference"]}] ↗</a>'
                        )
                    source_links = (
                        f'<span class="evidence-title-label">'
                        f'{title}</span> · '
                        + " ".join(reference_links)
                    )

                st.markdown(
                    f"""
                    <div class="evidence-item">
                        <div class="evidence-title">
                            {source_links}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def show_feedback(interaction_id: str) -> None:
    """Collect a rating and optional explanation for the answer."""

    feedback_panel = st.container(key="feedback_panel")

    with feedback_panel:
        st.caption("Was this answer useful?")
        action_row = st.container(
            horizontal=True,
            gap="small",
            vertical_alignment="center",
        )

        with action_row:
            selection = st.feedback(
                "thumbs",
                key=f"feedback-{interaction_id}",
                width="content",
            )

        if selection is not None:
            feedback = 1 if selection == 1 else -1

            if st.session_state.get("feedback") != feedback:
                record_feedback(interaction_id, feedback)
                st.session_state["feedback"] = feedback
                st.toast("Feedback recorded — thank you.")

        with action_row:
            with st.popover(
                "Add feedback note",
                type="tertiary",
                icon=":material/edit_note:",
                disabled=(
                    st.session_state.get("feedback") not in {-1, 1}
                ),
                help="Rate the answer before adding a note.",
                key=f"feedback-note-popover-{interaction_id}",
            ):
                with st.form(f"feedback-comment-form-{interaction_id}"):
                    feedback_comment = st.text_area(
                        "What worked or what could improve?",
                        height=100,
                        max_chars=1000,
                        key=f"feedback-comment-{interaction_id}",
                    )
                    comment_submitted = st.form_submit_button(
                        "Save note"
                    )

                if comment_submitted:
                    record_feedback(
                        interaction_id,
                        st.session_state["feedback"],
                        feedback_comment,
                    )
                    st.toast("Feedback note saved.")


st.markdown(
    """
    <div class="brand-lockup">
        <svg
            class="brand-mark"
            viewBox="0 0 48 48"
            role="img"
            aria-label="Supplement Evidence Lens logo"
        >
            <rect
                x="1"
                y="1"
                width="46"
                height="46"
                rx="13"
                fill="#245E4C"
            />
            <circle
                cx="21"
                cy="20"
                r="11"
                fill="none"
                stroke="#FFFFFF"
                stroke-width="2.5"
            />
            <path
                d="M29 28L38 37"
                fill="none"
                stroke="#FFFFFF"
                stroke-linecap="round"
                stroke-width="3"
            />
            <clipPath id="capsule-shape">
                <rect x="13" y="17" width="16" height="7" rx="3.5"/>
            </clipPath>
            <g clip-path="url(#capsule-shape)">
                <rect x="13" y="17" width="8" height="7" fill="#B9DDD0"/>
                <rect x="21" y="17" width="8" height="7" fill="#FFFFFF"/>
            </g>
            <rect
                x="13"
                y="17"
                width="16"
                height="7"
                rx="3.5"
                fill="none"
                stroke="#FFFFFF"
                stroke-width="1.25"
            />
            <path d="M21 17V24" stroke="#245E4C" stroke-width="1"/>
        </svg>
        <h1 class="hero-title">Supplement Evidence Lens</h1>
    </div>
    <p class="hero-copy">
        Evidence-grounded answers from official U.S., Canadian, and
        European supplement sources.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.form("question-form"):
    question = st.text_area(
        "Ask a question",
        value=st.session_state.get("question", ""),
        placeholder=(
            "Try asking:\n"
            "Does melatonin actually help with sleep?\n"
            "What are the risks of taking too much zinc?"
        ),
        height=110,
    )
    _, button_column = st.columns([4.1, 1.4])
    with button_column:
        submitted = st.form_submit_button(
            "Search Evidence  →",
            type="primary",
            use_container_width=False,
        )

if submitted:
    question = question.strip()

    if not question:
        st.warning("Enter a question before searching.")
    else:
        try:
            started_at = perf_counter()

            with st.spinner(
                "Searching official sources and reviewing the evidence..."
            ):
                rag = load_rag()
                result = rag.answer(question)

            interaction_id = record_interaction(
                result,
                perf_counter() - started_at,
                rag.model_name,
                rag.rag_version,
            )
            st.session_state["result"] = result
            st.session_state["interaction_id"] = interaction_id
            st.session_state["feedback"] = None

        except Exception as error:
            st.error(
                "The assistant could not complete this request. "
                "Check that Elasticsearch is running and that "
                "OPENAI_API_KEY is set in .env."
            )
            with st.expander("Technical details"):
                st.code(str(error))

if (
    "result" in st.session_state
    and "interaction_id" in st.session_state
):
    displayed_result = st.session_state["result"]
    st.divider()
    st.header("Answer")
    st.markdown(displayed_result["answer"])
    show_feedback(st.session_state["interaction_id"])
    show_contexts(displayed_result["contexts"])

st.markdown(
    """
    <div class="disclaimer">
        This tool summarizes official-source excerpts for informational
        purposes. It does not diagnose conditions, recommend treatment, or
        replace advice from a qualified health professional.
    </div>
    """,
    unsafe_allow_html=True,
)
