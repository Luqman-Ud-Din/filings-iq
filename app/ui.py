"""FR-9 — Streamlit UI.

A single-page chat over Apple's 10-K filings. Streams the answer and renders
citations beneath it. Reuses the same app.core.answer() path as the CLI and the
eval harness — no parallel logic.

    streamlit run app/ui.py
"""

import streamlit as st

from app.core import answer

st.set_page_config(page_title="Ask the 10-K", page_icon="📄")
st.title("📄 Ask the 10-K")
st.caption(
    "Grounded answers over Apple's three most recent 10-K filings — every claim "
    "cited, and a clean refusal when the answer isn't in the filings."
)

if "history" not in st.session_state:
    st.session_state.history = []

# Replay prior turns.
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("citations"):
            st.caption("Sources: " + "  ".join(f"`[{c}]`" for c in turn["citations"]))

question = st.chat_input("Ask about Apple's 10-K filings…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer = {"text": ""}

        def on_token(piece):
            buffer["text"] += piece
            placeholder.markdown(buffer["text"])

        try:
            result = answer(question, stream=False, trace=True, on_token=on_token)
        except RuntimeError as exc:
            placeholder.error(str(exc))
            st.stop()

        placeholder.markdown(result["answer"])
        citations = [] if result["refused"] else result["citations"]
        if citations:
            st.caption("Sources: " + "  ".join(f"`[{c}]`" for c in citations))

    st.session_state.history.append(
        {"role": "assistant", "content": result["answer"], "citations": citations}
    )
