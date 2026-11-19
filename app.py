"""Streamlit dashboard: run the pipeline, review ranked promos, inspect one, export.

    streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src.exporter import promo_record, segments_str, to_csv, to_json, to_markdown
from src.llm_provider import PROVIDERS
from src.pipeline import list_saved, load_saved, run_pipeline
from src.promo_builder import promo_timeline, quote_matches
from src.screenplay_parser import fmt_time

st.set_page_config(page_title="Kuku Promo Finder", page_icon="🎬", layout="wide")

KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai_compatible": "LLM_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


# --------------------------------------------------------------------------- sidebar
def sidebar():
    st.sidebar.title("🎬 Promo Finder")
    saved = list_saved()
    if saved:
        st.sidebar.subheader("Saved runs")
        choice = st.sidebar.selectbox("Show", saved, index=0, label_visibility="collapsed")
        if st.sidebar.button("Load results", width="stretch") or "result" not in st.session_state:
            st.session_state.result = load_saved(choice)

    st.sidebar.subheader("Process a show")
    mode = st.sidebar.radio("Input", ["Folder path", "Upload .zip"], horizontal=True)
    input_path = None
    if mode == "Folder path":
        default = next((str(p) for p in Path(config.DEFAULT_INPUT_DIR).glob("*") if p.is_dir()),
                       str(config.DEFAULT_INPUT_DIR))
        input_path = st.sidebar.text_input("Folder with episode .txt files", default)
    else:
        up = st.sidebar.file_uploader("Zip of episode scripts", type="zip")
        if up:
            tmp = Path(tempfile.gettempdir()) / "kuku_promo_uploads"
            tmp.mkdir(exist_ok=True)
            input_path = str(tmp / up.name)
            Path(input_path).write_bytes(up.getvalue())

    names = list(PROVIDERS)
    provider = st.sidebar.selectbox("LLM provider", names,
                                    index=names.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in names else 0,
                                    help="'replay' re-runs a previous execution from cache only (no API calls).")
    model = st.sidebar.text_input("Model (blank = provider default)", config.LLM_MODEL)
    env = KEY_ENV.get(provider)
    if env:
        st.sidebar.caption(("✅ " if os.getenv(env) else "⚠️ missing ") + env)
    episodes = st.sidebar.text_input("Episodes (blank = all, e.g. 1-5,9)", "")
    cross = st.sidebar.checkbox("Cross-episode montages", True)
    blind = st.sidebar.number_input("Blind-viewer test (-1 = all, 0 = off, N = top N)", -1, 500, -1)
    if st.sidebar.button("▶ Run pipeline", type="primary", width="stretch", disabled=not input_path):
        run(input_path, provider, model, episodes, cross, int(blind))


def run(input_path, provider, model, episodes, cross, blind):
    from run_pipeline import parse_range
    bar = st.progress(0.0, text="starting")
    log = st.status("Processing…", expanded=True)

    def progress(stage, done, total, msg):
        frac = {"ingest": 0.05, "episodes": 0.05 + 0.75 * done / max(total, 1), "cross_episode": 0.82,
                "rank": 0.85, "blind_test": 0.85 + 0.14 * done / max(total, 1), "done": 1.0}.get(stage, 0.5)
        bar.progress(min(frac, 1.0), text=f"{stage}: {msg}")
        if "FAILED" in msg:
            log.error(msg)
        elif stage in ("ingest", "cross_episode", "rank", "done"):
            log.write(msg)

    try:
        res = run_pipeline(input_path, provider, model, parse_range(episodes), cross, blind, progress)
    except Exception as exc:  # surfaced to the user, never swallowed
        log.update(label="Failed", state="error")
        st.error(f"Pipeline failed: {exc}")
        return
    log.update(label="Done", state="complete")
    st.session_state.result = load_saved(res["meta"]["slug"])
    if res["meta"]["episode_errors"]:
        st.warning(f"LLM step failed for: {res['meta']['episode_errors']}")


# --------------------------------------------------------------------------- views
def overview(res):
    meta, rep, ing = res["meta"], res["report"], res["ingest"]
    c = meta["counts"]
    st.header(meta["show"])
    cols = st.columns(5)
    cols[0].metric("Episodes processed", len(meta["episodes_processed"]))
    cols[1].metric("Candidate promos", c["candidates"])
    cols[2].metric("Valid & ranked", c["valid_ranked"])
    cols[3].metric("Rejected", c["rejected"])
    cols[4].metric("Duplicates", c["duplicates"])
    st.caption(f"Provider `{meta['provider']}` · models {', '.join(meta.get('models_reported', []))} · "
               f"prompt {meta['prompt_version']} · run {meta['run_utc']} · {meta['elapsed_s']} s · "
               f"LLM calls {meta['llm_calls']} ({meta['llm_calls_from_cache']} from cache)")
    if meta.get("episode_errors"):
        st.error(f"LLM failures: {meta['episode_errors']}")
    a, b = st.columns(2)
    with a:
        st.subheader("Input")
        if ing:
            s = ing["stats"]
            st.write(f"{s['episodes']} episodes · {s['blocks']} timed blocks · runtime "
                     f"{s['total_runtime_s'] / 60:.1f} min · episodes {s['min_episode_s']:.0f}–{s['max_episode_s']:.0f}s")
            st.write(f"Block types: {s['block_kinds']}")
            with st.expander(f"Parser warnings ({len(ing['warnings'])}) / errors ({len(ing['errors'])})"):
                st.write(ing["errors"] or "no errors")
                st.write(ing["warnings"] or "no warnings")
    with b:
        st.subheader("Rejections (deterministic validation)")
        rr = rep.get("rejection_reasons", {})
        if rr:
            st.bar_chart(pd.Series(rr, name="count"))
        else:
            st.write("none")
    st.subheader("Scoring formula")
    st.write("final = 10 × Σ weight × LLM score (1–10)  −  deterministic penalties")
    w = pd.DataFrame({"weight": meta["weights"]}).T
    st.dataframe(w, width="stretch")
    st.caption("Penalties: " + ", ".join(f"{k} −{v}" for k, v in meta["penalties"].items()))


def ranked_table(res):
    ranked = [p for p in res["promos"] if p.rank]
    st.subheader(f"{len(ranked)} ranked promos")
    df = pd.DataFrame([{
        "rank": p.rank, "id": p.promo_id, "score": p.final_score, "secs": round(p.duration),
        "episodes": ", ".join(f"E{e}" for e in p.episodes), "segments": segments_str(p),
        "hook": p.hook_text, "cliffhanger": p.cliffhanger_text, "why it works": p.why_it_works,
    } for p in ranked])
    if df.empty:
        st.info("No valid promos.")
        return
    q = st.text_input("Filter (episode, text…)", "")
    if q:
        df = df[df.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(df, width="stretch", hide_index=True, height=600,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100,
                                                                         format="%.1f")})


def detail(res):
    promos = res["promos"]
    eps = res["episodes"]
    label = {p.promo_id: (f"#{p.rank} · " if p.rank else f"[{p.status}] ") + f"{p.promo_id} · {p.final_score:.1f} · {p.title}"
             for p in promos}
    pid = st.selectbox("Promo", [p.promo_id for p in promos], format_func=lambda k: label[k])
    p = next(x for x in promos if x.promo_id == pid)
    st.subheader(f"{label[pid]}")
    status_color = {"valid": "green", "duplicate": "orange", "rejected": "red"}.get(p.status, "gray")
    st.markdown(f"**Status:** :{status_color}[{p.status}]" + (f" (duplicate of {p.duplicate_of})" if p.duplicate_of else "")
                + f" · **{p.duration:.1f}s** · {len(p.segments)} segment(s)")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Hook** ({p.hook_offset_s}s in): {p.hook_text}")
        st.markdown(f"**Cliffhanger** (cut {p.cliff_end_gap_s}s later): {p.cliffhanger_text}")
        st.markdown(f"**Open question:** {p.open_question or '—'}")
        st.markdown(f"**Why it works:** {p.why_it_works}")
        if p.rank_reason:
            st.markdown(f"**Why this rank:** {p.rank_reason}")
        if p.context_gap:
            st.markdown(f"**Context a new viewer may lack (LLM self-report):** {p.context_gap}")
        st.dataframe(pd.DataFrame([{"#": i + 1, "episode": s.episode, "start": fmt_time(s.start),
                                    "end": fmt_time(s.end), "secs": s.duration,
                                    "blocks": f"{s.first_block}–{s.last_block}"}
                                   for i, s in enumerate(p.segments)]), hide_index=True, width="stretch")
    with c2:
        contrib = {k: round(config.WEIGHTS[k] * p.llm_scores.get(k, 0) * 10, 1) for k in config.WEIGHTS}
        st.markdown(f"**Score {p.final_score:.1f}** = weighted {p.weighted_score:.1f} − penalties "
                    f"{sum(p.penalties.values()):g}")
        st.dataframe(pd.DataFrame({"LLM score (1-10)": p.llm_scores, "points": contrib}),
                     width="stretch")
        if p.penalties:
            st.write({k: f"−{v:g}" for k, v in p.penalties.items()})
        st.markdown("**Validation**")
        if not p.issues:
            st.success("All deterministic checks passed")
        for i in p.issues:
            (st.error if i.severity == "error" else st.warning)(f"`{i.code}` {i.message}")
        if p.blind_test:
            with st.expander("Blind-viewer test (independent LLM, transcript only)", expanded=True):
                bt = p.blind_test
                st.write(f"Understood **{bt.get('understood_score')}**/10 · first 3s stop-scroll "
                         f"**{bt.get('first_3s_stop_scroll')}**/10 · want next **{bt.get('want_next_episode')}**/10")
                st.write(f"_What they saw:_ {bt.get('what_happens')}")
                st.write(f"_Who is who:_ {bt.get('who_is_who')}")
                st.write(f"_Left asking:_ {bt.get('open_question')}")
                if bt.get("confusing_points"):
                    st.write("_Confusing:_ " + "; ".join(bt["confusing_points"]))

    st.markdown("### Original screenplay")
    show_ctx = st.checkbox("Show 10 s of surrounding context (greyed)", False)
    if not eps:
        st.info("Parsed episodes not found for this run.")
        return
    tl = promo_timeline(p, eps)
    in_promo = {(seg.episode, b.idx): (ps, pe) for ps, pe, seg, b in tl}
    for i, seg in enumerate(p.segments, start=1):
        ep = eps[seg.episode]
        st.markdown(f"**Segment {i} — Episode {seg.episode}, {fmt_time(seg.start)} → {fmt_time(seg.end)}**")
        rows = []
        lo, hi = (seg.start - 10, seg.end + 10) if show_ctx else (seg.start, seg.end)
        for b in ep.blocks:
            if not b.timed or b.end <= lo or b.start >= hi:
                continue
            key = (seg.episode, b.idx)
            inside = key in in_promo
            if not inside and not show_ctx:
                continue
            mark = ""
            if inside and quote_matches(p.hook_text, b.text):
                mark = "🪝 HOOK"
            elif inside and quote_matches(p.cliffhanger_text, b.text):
                mark = "⏸ CLIFF"
            rows.append({"": mark if inside else "· outside", "promo t": f"{in_promo[key][0]:.1f}s" if inside else "",
                         "episode time": f"{fmt_time(b.start)}–{fmt_time(b.end)}", "type": b.kind,
                         "label": b.label, "text": b.text})
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(df.style.apply(lambda r: ["color: #999" if r[""] == "· outside" else "" for _ in r], axis=1),
                         hide_index=True, width="stretch")


def others(res):
    rest = [p for p in res["promos"] if not p.rank]
    st.subheader(f"Rejected ({sum(p.status == 'rejected' for p in rest)}) and duplicates "
                 f"({sum(p.status == 'duplicate' for p in rest)})")
    st.dataframe(pd.DataFrame([{
        "id": p.promo_id, "status": p.status, "score": p.final_score, "secs": p.duration,
        "segments": segments_str(p), "dup of": p.duplicate_of or "",
        "errors": " | ".join(i.message for i in p.issues if i.severity == "error"), "hook": p.hook_text,
    } for p in rest]), hide_index=True, width="stretch")


def quality(res):
    base = Path(res["dirs"]["base"]) / "evaluation"
    md = base / "quality_report.md"
    st.markdown(md.read_text(encoding="utf-8") if md.exists() else "No report yet.")
    manual = base / "manual_review.md"
    if manual.exists():
        with st.expander("Manual review notes"):
            st.markdown(manual.read_text(encoding="utf-8"))


def export(res):
    promos, meta = res["promos"], res["meta"]
    slug = meta["slug"]
    st.download_button("⬇ promos.json", to_json(promos, meta), f"{slug}_promos.json", "application/json")
    st.download_button("⬇ promos.csv", to_csv(promos).encode("utf-8-sig"), f"{slug}_promos.csv", "text/csv")
    st.download_button("⬇ ranked_promos.md", to_markdown(promos, res["episodes"], meta["show"]),
                       f"{slug}_ranked_promos.md", "text/markdown")
    st.caption(f"Files are also written to `{res['dirs']['base']}`")
    with st.expander("Preview JSON (top 3)"):
        st.json([promo_record(p) for p in promos if p.rank][:3])


def main():
    sidebar()
    res = st.session_state.get("result")
    if not res:
        st.title("Kuku Promo Finder")
        st.info("Run the pipeline from the sidebar (or load a saved run). Configure the LLM in `.env` — "
                "see `.env.example`.")
        return
    tabs = st.tabs(["Overview", "Ranked promos", "Promo detail", "Rejected / duplicates", "Quality", "Export"])
    with tabs[0]:
        overview(res)
    with tabs[1]:
        ranked_table(res)
    with tabs[2]:
        detail(res)
    with tabs[3]:
        others(res)
    with tabs[4]:
        quality(res)
    with tabs[5]:
        export(res)


main()
