#!/usr/bin/env python3
"""Introduction and documentation page for the inspection-planning app."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
CONTENT_DIR = APP_DIR / "content"


def data_uri(path: Path) -> str:
    """Return a browser-safe data URI for a local image asset."""

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def editable_markdown(filename: str, fallback: str) -> str:
    """Load page copy from a file that can be edited directly on GitHub."""

    path = CONTENT_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else fallback


logo = data_uri(ASSET_DIR / "university_of_alberta_logo.svg")
mohammadali = data_uri(ASSET_DIR / "mohammadali_ameri.png")
yong = data_uri(ASSET_DIR / "yong_li.png")

st.markdown(
    """
    <style>
    :root {
        --ua-green: #275d38;
        --ua-gold: #ffdb05;
        --ink: #14251c;
        --muted: #5b6961;
        --line: #dce5de;
    }
    .stApp { background: linear-gradient(180deg, #f1f5ef 0, #ffffff 30rem); }
    [data-testid="stHeader"] { background: rgba(241, 245, 239, .92); }
    [data-testid="stSidebar"] { background: #f3f6f2; }
    .block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 4rem; }
    .brand-hero {
        position: relative; overflow: hidden; padding: 1.5rem 1.7rem 1.65rem;
        border: 1px solid #d4e0d7; border-top: 7px solid var(--ua-green);
        border-radius: 20px; background: rgba(255,255,255,.95);
        box-shadow: 0 14px 36px rgba(20, 49, 32, .08);
    }
    .brand-hero::after {
        content: ""; position: absolute; width: 180px; height: 180px;
        right: -70px; top: -85px; border-radius: 50%; background: var(--ua-gold);
        opacity: .78;
    }
    .brand-logo { width: 245px; max-width: 58%; height: auto; display: block; margin-bottom: 1.2rem; }
    .eyebrow { color: var(--ua-green); font-weight: 800; letter-spacing: .11em;
        text-transform: uppercase; font-size: .77rem; }
    .brand-hero h1 { color: var(--ink); font-family: Georgia, serif; font-weight: 500;
        font-size: clamp(2rem, 4vw, 3.45rem); line-height: 1.05; margin: .38rem 0 .65rem; }
    .brand-hero p { color: var(--muted); font-size: 1.03rem; max-width: 760px; margin: 0; }
    .credit-line { color: var(--ink); font-size: .92rem; font-weight: 700; margin-top: 1rem; }
    .people-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem; margin: 1.15rem 0 1.8rem; }
    .person-card { display: flex; align-items: center; gap: 1rem; padding: 1rem;
        background: white; border: 1px solid var(--line); border-radius: 16px;
        box-shadow: 0 6px 18px rgba(20, 49, 32, .05); }
    .person-card img { width: 104px; height: 104px; flex: 0 0 104px; object-fit: cover;
        border-radius: 50%; border: 4px solid white; outline: 3px solid var(--ua-green); }
    .person-card h3 { color: var(--ink); font-size: 1.15rem; margin: 0 0 .25rem; }
    .person-card p { color: var(--muted); margin: 0; font-size: .9rem; line-height: 1.35; }
    .section-label { color: var(--ua-green); font-size: .78rem; font-weight: 800;
        letter-spacing: .1em; text-transform: uppercase; margin-top: 1.4rem; }
    .research-note { margin-top: 1.7rem; padding: .9rem 1rem; background: #eef4ef;
        border-left: 4px solid var(--ua-green); color: #405047; border-radius: 6px; }
    @media (max-width: 700px) {
        .people-grid { grid-template-columns: 1fr; }
        .person-card img { width: 86px; height: 86px; flex-basis: 86px; }
        .brand-hero { padding: 1.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="brand-hero">
      <img class="brand-logo" src="{logo}" alt="University of Alberta">
      <div class="eyebrow">Reliability-based pipeline integrity planning</div>
      <h1>Pipeline Inspection Scheduling</h1>
      <p>Interactive decision support for comparing fixed inspection intervals across
      multiple pipe joints under deterioration, inspection uncertainty, repair actions,
      failure risk, and life-cycle cost.</p>
      <div class="credit-line">Developed by Mohammadali Ameri and Yong Li</div>
    </section>

    <section class="people-grid" aria-label="Application developers">
      <article class="person-card">
        <img src="{mohammadali}" alt="Mohammadali Ameri">
        <div><h3>Mohammadali Ameri</h3><p>Researcher and developer<br>University of Alberta</p></div>
      </article>
      <article class="person-card">
        <img src="{yong}" alt="Yong Li">
        <div><h3>Yong Li</h3><p>Supervisor and co-developer<br>University of Alberta</p></div>
      </article>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Project abstract</div>', unsafe_allow_html=True)
st.markdown(
    editable_markdown(
        "ABSTRACT.md",
        "Add the project abstract in `content/ABSTRACT.md`.",
    )
)

st.page_link(
    "inspection_planner.py",
    label="Open the inspection scheduling tool",
    icon=":material/arrow_forward:",
    use_container_width=True,
)

st.divider()
st.markdown(
    editable_markdown(
        "DOCUMENTATION.md",
        "Add the application documentation in `content/DOCUMENTATION.md`.",
    )
)

st.markdown(
    """
    <div class="research-note"><strong>Research-use notice.</strong> This application is a
    decision-support prototype. Its results require engineering review and do not replace
    ILI vendor validation, applicable codes, regulatory requirements, or an operator's
    integrity-management procedures.</div>
    """,
    unsafe_allow_html=True,
)
