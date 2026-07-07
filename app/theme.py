"""
theme.py
=========
Centralized visual identity for the JusticeLens AI Streamlit application.

Design intent: this is a government-facing decision-support tool, not a
consumer product, so the visual language borrows from official case
records and gazettes rather than generic dashboard/SaaS styling --
navy-and-gold, a serif display face for headers, and a recurring
left-accent "case file" card motif used for every metric/insight panel
(color-coded by disparity tier: gold for underserved, teal for
adequately-served). Typography is IBM Plex (Serif + Sans + Mono), chosen
deliberately because this system is built on IBM watsonx.ai/Granite --
the type family itself reinforces what the product is.

Every page in ``app/`` calls :func:`apply_page_config` first, then
:func:`inject_global_css` once per page render, then composes the page
body using the helper components below (``render_app_header``,
``metric_card``, ``tier_badge``, ``section_eyebrow``, ``style_plotly_fig``)
rather than hand-rolling ad-hoc HTML/CSS, so visual consistency is
enforced structurally rather than by convention alone.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

# --------------------------------------------------------------------------- #
# Palette -- named, deliberate, government-record inspired (not the default
# warm-cream/terracotta or near-black/neon AI-generated look).
# --------------------------------------------------------------------------- #

INK_NAVY = "#14213D"       #: Primary dark -- sidebar, headers, dark surfaces
DEEP_SLATE = "#24344D"     #: Secondary dark panels
STEEL_BLUE = "#3E6488"     #: Links, secondary accents, neutral data series
STATUTE_GOLD = "#C9A24B"   #: Primary accent -- CTAs, highlights, "official seal" gold
VERDICT_TEAL = "#2E7D6B"   #: Positive / "Adequately Served" signal
OXBLOOD = "#8C3A3A"        #: Risk / "Underserved" signal
PAPER = "#F7F6F2"          #: Page background -- warm paper, not stark white
CARD_WHITE = "#FFFFFF"     #: Card/panel surface on top of PAPER
INK_TEXT = "#1F2430"       #: Primary body text
MUTED_TEXT = "#6B7280"     #: Secondary/caption text
BORDER = "#E3E0D8"         #: Hairline borders on PAPER

#: Tier -> color mapping used consistently across every chart, badge, and
#: card in the app. Kept to the two model classes
#: (``config.ML_CLASS_NAMES``) plus a neutral "unknown" fallback.
TIER_COLORS = {
    "Underserved": OXBLOOD,
    "Adequately Served": VERDICT_TEAL,
    "unknown": MUTED_TEXT,
}

#: Ordered categorical color sequence for multi-series Plotly charts
#: (state comparisons, model comparisons, etc.), built from the palette
#: above rather than Plotly's default rainbow sequence.
CATEGORICAL_SEQUENCE = [
    STEEL_BLUE,
    STATUTE_GOLD,
    VERDICT_TEAL,
    OXBLOOD,
    DEEP_SLATE,
    "#8A9BA8",
]

#: Diverging sequence for choropleth-style / intensity charts (low
#: disparity -> high disparity).
DISPARITY_SCALE = [
    [0.0, VERDICT_TEAL],
    [0.5, STATUTE_GOLD],
    [1.0, OXBLOOD],
]

FONT_DISPLAY = "'IBM Plex Serif', Georgia, serif"
FONT_BODY = "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif"
FONT_MONO = "'IBM Plex Mono', 'Courier New', monospace"

APP_TITLE = "JusticeLens AI"
APP_TAGLINE = "Tele-Law Legal Access Disparity Intelligence"


def apply_page_config(page_title: str, page_icon: str = "⚖️") -> None:
    """Set the Streamlit page configuration consistently across every
    page. Must be the first Streamlit call on any page.

    Args:
        page_title: Title shown in the browser tab, prefixed with the app
            name for consistency.
        page_icon: Emoji or path used as the browser tab icon.
    """
    st.set_page_config(
        page_title=f"{page_title} | {APP_TITLE}",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_global_css() -> None:
    """Inject the app's global CSS: font imports, color variables, card/
    badge styling, sidebar branding, and Streamlit default-chrome
    overrides. Safe to call on every page (idempotent from the browser's
    perspective; Streamlit re-renders the ``<style>`` block each run).
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_BODY};
            color: {INK_TEXT};
        }}

        .stApp {{
            background-color: {PAPER};
        }}

        h1, h2, h3 {{
            font-family: {FONT_DISPLAY};
            color: {INK_NAVY};
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {INK_NAVY};
        }}
        section[data-testid="stSidebar"] * {{
            color: {PAPER} !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(247, 246, 242, 0.2);
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            background-color: {INK_NAVY};
            color: {PAPER};
            border: 1px solid {INK_NAVY};
            border-radius: 4px;
            font-family: {FONT_BODY};
            font-weight: 500;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {STATUTE_GOLD};
            border-color: {STATUTE_GOLD};
            color: {INK_NAVY};
        }}

        /* Section eyebrow label */
        .jl-eyebrow {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {STEEL_BLUE};
            margin-bottom: 0.15rem;
        }}

        /* Case-file card: the app's signature recurring device -- a
           left-accent-bordered panel, color-coded by meaning (tier,
           status, emphasis) rather than decorative. */
        .jl-card {{
            background-color: {CARD_WHITE};
            border: 1px solid {BORDER};
            border-left: 5px solid {STEEL_BLUE};
            border-radius: 6px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }}
        .jl-card.jl-card--gold {{ border-left-color: {STATUTE_GOLD}; }}
        .jl-card.jl-card--teal {{ border-left-color: {VERDICT_TEAL}; }}
        .jl-card.jl-card--oxblood {{ border-left-color: {OXBLOOD}; }}
        .jl-card.jl-card--navy {{ border-left-color: {INK_NAVY}; }}

        .jl-card h4 {{
            font-family: {FONT_DISPLAY};
            font-size: 1.05rem;
            margin: 0 0 0.3rem 0;
            color: {INK_NAVY};
        }}
        .jl-card p {{
            margin: 0;
            color: {MUTED_TEXT};
            font-size: 0.9rem;
        }}

        /* Metric number display */
        .jl-metric-value {{
            font-family: {FONT_MONO};
            font-size: 2.1rem;
            font-weight: 500;
            color: {INK_NAVY};
            line-height: 1.1;
        }}
        .jl-metric-label {{
            font-family: {FONT_BODY};
            font-size: 0.82rem;
            color: {MUTED_TEXT};
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        /* Tier badge (pill) */
        .jl-badge {{
            display: inline-block;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-family: {FONT_BODY};
            font-size: 0.78rem;
            font-weight: 600;
            color: {CARD_WHITE};
        }}

        /* AI provenance tag -- distinguishes Granite-generated text from
           deterministic template fallback, used throughout the Policy
           Room and report views. */
        .jl-provenance {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            color: {MUTED_TEXT};
            border-top: 1px dashed {BORDER};
            margin-top: 0.6rem;
            padding-top: 0.4rem;
        }}

        /* Hide Streamlit's default "Made with Streamlit" footer for a
           cleaner, more official presentation. */
        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Render the consistent JusticeLens AI brand block at the top of the
    sidebar. Call once per page, immediately after :func:`inject_global_css`.
    """
    st.sidebar.markdown(
        f"""
        <div style="padding: 0.4rem 0 1rem 0;">
            <div style="font-family: {FONT_DISPLAY}; font-size: 1.35rem; font-weight: 700; color: {PAPER};">
                ⚖️ {APP_TITLE}
            </div>
            <div style="font-family: {FONT_BODY}; font-size: 0.78rem; color: {STATUTE_GOLD}; margin-top: 0.1rem;">
                {APP_TAGLINE}
            </div>
        </div>
        <hr/>
        """,
        unsafe_allow_html=True,
    )


def section_eyebrow(text: str) -> None:
    """Render a small-caps monospace "eyebrow" label above a section
    heading -- an official-document-style structural device used
    throughout the app instead of decorative dividers.

    Args:
        text: Short label text (rendered upper-cased via CSS).
    """
    st.markdown(f'<div class="jl-eyebrow">{text}</div>', unsafe_allow_html=True)


def render_app_header(title: str, subtitle: Optional[str] = None, eyebrow: Optional[str] = None) -> None:
    """Render a consistent page-level header: optional eyebrow, serif
    title, optional muted subtitle.

    Args:
        title: Main page title.
        subtitle: Optional one-line description shown beneath the title.
        eyebrow: Optional small-caps label shown above the title.
    """
    if eyebrow:
        section_eyebrow(eyebrow)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(
            f'<p style="color:{MUTED_TEXT}; font-size:0.95rem; margin-top:-0.6rem;">{subtitle}</p>',
            unsafe_allow_html=True,
        )


def tier_badge_html(tier: str) -> str:
    """Build the HTML for a color-coded tier badge (pill).

    Args:
        tier: One of ``config.ML_CLASS_NAMES`` (e.g. "Underserved",
            "Adequately Served"), or any other string (rendered neutrally).

    Returns:
        An HTML string. Pass to ``st.markdown(..., unsafe_allow_html=True)``.
    """
    color = TIER_COLORS.get(tier, TIER_COLORS["unknown"])
    return f'<span class="jl-badge" style="background-color:{color};">{tier}</span>'


def metric_card(label: str, value: str, card_variant: str = "navy", note: Optional[str] = None) -> None:
    """Render a single "case file" metric card -- the app's signature
    recurring component for headline numbers.

    Args:
        label: Short uppercase-style label (styled via CSS).
        value: The metric value, already formatted as a display string
            (e.g. "23.4%", "1,204").
        card_variant: One of "navy", "gold", "teal", "oxblood" -- selects
            the left-accent border color, used to encode meaning (e.g.
            "oxblood" for a risk metric).
        note: Optional small caption shown beneath the value.
    """
    note_html = f"<p>{note}</p>" if note else ""
    st.markdown(
        f"""
        <div class="jl-card jl-card--{card_variant}">
            <div class="jl-metric-label">{label}</div>
            <div class="jl-metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, body: str, card_variant: str = "navy") -> None:
    """Render a "case file" card for a titled block of text (narrative
    summaries, recommendations, etc.), consistent with :func:`metric_card`.

    Args:
        title: Card heading.
        body: Card body text (may include simple HTML).
        card_variant: One of "navy", "gold", "teal", "oxblood".
    """
    st.markdown(
        f"""
        <div class="jl-card jl-card--{card_variant}">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def provenance_tag(is_ai_generated: bool, model_id: str) -> None:
    """Render the small provenance tag distinguishing Granite-generated
    narration from deterministic template fallback, shown beneath every
    AI-generated block in the app.

    Args:
        is_ai_generated: Whether the text was produced by a live watsonx.ai
            /Granite call.
        model_id: The model identifier, or ``"template_fallback"``.
    """
    source = f"Generated by IBM Granite ({model_id}) via watsonx.ai" if is_ai_generated else (
        "Generated by deterministic template -- watsonx.ai was unavailable at generation time"
    )
    icon = "🤖" if is_ai_generated else "📋"
    st.markdown(f'<div class="jl-provenance">{icon} {source}</div>', unsafe_allow_html=True)


def style_plotly_fig(fig, height: int = 420):
    """Apply the app's consistent Plotly styling to a figure: font family,
    paper/plot background, color sequence, and margins.

    Args:
        fig: A ``plotly.graph_objects.Figure`` instance.
        height: Figure height in pixels.

    Returns:
        The same figure, mutated in place and returned for convenience.
    """
    fig.update_layout(
        height=height,
        font=dict(family=FONT_BODY, color=INK_TEXT, size=13),
        title_font=dict(family=FONT_DISPLAY, color=INK_NAVY, size=18),
        paper_bgcolor=PAPER,
        plot_bgcolor=CARD_WHITE,
        colorway=CATEGORICAL_SEQUENCE,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig
