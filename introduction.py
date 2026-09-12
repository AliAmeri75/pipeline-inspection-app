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
ili_illustration = data_uri(ASSET_DIR / "ili_pipeline.svg")

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
    .planner-cta {
        display: grid; grid-template-columns: 86px minmax(0, 1fr) 260px;
        align-items: center; gap: 1.25rem; margin: 1.45rem 0 1.2rem;
        padding: 1.15rem 1.3rem; color: white !important; text-decoration: none !important;
        background: linear-gradient(120deg, #1f4d30 0%, var(--ua-green) 58%, #173d27 100%);
        border: 3px solid var(--ua-gold); border-radius: 18px;
        box-shadow: 0 14px 30px rgba(20, 49, 32, .2);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .planner-cta:hover { transform: translateY(-3px); box-shadow: 0 18px 38px rgba(20, 49, 32, .28); }
    .calendar-icon {
        display: grid; place-items: center; width: 76px; height: 76px;
        color: var(--ua-green); background: var(--ua-gold); border-radius: 18px;
        box-shadow: inset 0 0 0 3px rgba(255,255,255,.55);
    }
    .calendar-icon svg { width: 46px; height: 46px; }
    .cta-kicker { display: block; color: #fff3a1; font-size: .76rem; font-weight: 900;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .18rem; }
    .cta-title { display: block; color: white; font-size: clamp(1.25rem, 2.5vw, 1.72rem);
        font-weight: 900; line-height: 1.13; }
    .cta-detail { display: block; color: #e2eee5; font-size: .92rem; margin-top: .36rem; }
    .cta-arrow { color: var(--ua-gold); font-size: 1.35em; padding-left: .25rem; }
    .ili-picture { width: 100%; max-height: 118px; object-fit: contain; }
    .research-note { margin-top: 1.7rem; padding: .9rem 1rem; background: #eef4ef;
        border-left: 4px solid var(--ua-green); color: #405047; border-radius: 6px; }
    @media (max-width: 700px) {
        .people-grid { grid-template-columns: 1fr; }
        .person-card img { width: 86px; height: 86px; flex-basis: 86px; }
        .brand-hero { padding: 1.2rem; }
        .planner-cta { grid-template-columns: 64px minmax(0, 1fr); gap: .85rem; padding: 1rem; }
        .calendar-icon { width: 58px; height: 58px; border-radius: 14px; }
        .calendar-icon svg { width: 34px; height: 34px; }
        .ili-picture { grid-column: 1 / -1; max-height: 90px; }
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
        <div><h3>Yong Li</h3><p>Associate Professor and co-developer<br>University of Alberta</p></div>
      </article>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <a class="planner-cta" href="inspection_planner" target="_self"
       aria-label="Open the inspection scheduling application">
      <span class="calendar-icon" aria-hidden="true">
        <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="7" y="10" width="34" height="31" rx="5" fill="white" stroke="currentColor" stroke-width="3"/>
          <path d="M7 19h34M16 6v8M32 6v8" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
          <path d="m16 30 5 5 11-12" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </span>
      <span>
        <span class="cta-kicker">Start your analysis</span>
        <strong class="cta-title">Open the Inspection Scheduling App<span class="cta-arrow">→</span></strong>
        <span class="cta-detail">Define the pipe joints, compare inspection intervals, and review the results.</span>
      </span>
      <img class="ili-picture" src="{ili_illustration}"
           alt="Illustration of an inline inspection tool inside a pipeline">
    </a>
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
