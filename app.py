"""PhaseShift · Frequency-domain image steganography · Streamlit UI."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import streamlit as st

from phaseshift.engine import (
    PhaseShiftError,
    capacity_bits,
    encode_message,
    extract_message,
    load_grayscale_image,
    max_payload_bytes,
)
from phaseshift.visuals import difference_figure, image_figure, spectrum_figure

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhaseShift",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Reset & App background ── */
.stApp { background-color: #111318 !important; }
.stApp > header { background-color: #111318 !important; border-bottom: 1px solid #2a2f3a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #13161c !important; border-right: 1px solid #2a2f3a !important; }
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Sidebar brand ── */
.ps-sidebar-brand { display: flex; align-items: center; gap: 10px; padding: 20px 16px 16px; border-bottom: 1px solid #2a2f3a; margin-bottom: 8px; }
.ps-logo { width: 32px; height: 32px; background: #3b82f6; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800; color: white; letter-spacing: -0.5px; flex-shrink: 0; }
.ps-brand-name { font-size: 15px; font-weight: 700; color: #e8eaed; line-height: 1.2; }
.ps-brand-sub { font-size: 10.5px; color: #5a6170; line-height: 1.4; }

/* ── Radio navigation ── */
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label { padding: 8px 12px !important; border-radius: 7px !important; color: #9aa0b2 !important; font-size: 13.5px !important; cursor: pointer !important; transition: all 0.15s !important; }
[data-testid="stSidebar"] .stRadio label:hover { background: #1e2229 !important; color: #e8eaed !important; }
[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
[data-testid="stSidebar"] .stRadio input:checked + div { background: rgba(59,130,246,0.12) !important; color: #3b82f6 !important; border-radius: 7px; }

/* ── Main content area ── */
.main .block-container { padding-top: 28px !important; padding-left: 32px !important; padding-right: 32px !important; max-width: 1200px; }

/* ── Page header ── */
.ps-page-header { padding-bottom: 20px; border-bottom: 1px solid #2a2f3a; margin-bottom: 24px; }
.ps-page-header h1 { font-size: 22px !important; font-weight: 700 !important; color: #e8eaed !important; margin: 0 0 4px 0 !important; letter-spacing: -0.3px; }
.ps-page-header p { font-size: 13.5px; color: #9aa0b2; margin: 0; }

/* ── Cards ── */
.ps-card { background: #1a1d23; border: 1px solid #2a2f3a; border-radius: 10px; padding: 16px 18px; }
.ps-card h4 { font-size: 13px; font-weight: 600; color: #e8eaed; margin: 0 0 8px 0; }
.ps-card p { font-size: 12.5px; color: #9aa0b2; margin: 0; line-height: 1.6; }

/* ── Pipeline visualization ── */
.ps-pipeline { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
.ps-pipeline-step { background: #1a1d23; border: 1px solid #2a2f3a; border-radius: 8px; padding: 8px 14px; font-size: 12px; font-weight: 500; color: #e8eaed; white-space: nowrap; }
.ps-pipeline-step.accent { border-color: #3b82f6; color: #3b82f6; background: rgba(59,130,246,0.08); }
.ps-pipeline-arrow { color: #2a2f3a; font-size: 16px; }

/* ── Section labels ── */
.ps-section-label { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; color: #5a6170; text-transform: uppercase; margin-bottom: 10px; }

/* ── Metric cards ── */
.ps-metric-row { display: flex; gap: 12px; }
.ps-metric-card { flex: 1; background: #1a1d23; border: 1px solid #2a2f3a; border-radius: 9px; padding: 14px 16px; }
.ps-metric-card .ps-metric-label { font-size: 10.5px; color: #5a6170; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.ps-metric-card .ps-metric-value { font-size: 20px; font-weight: 700; color: #e8eaed; font-family: 'JetBrains Mono', monospace; line-height: 1.2; }
.ps-metric-card .ps-metric-sub { font-size: 11px; color: #5a6170; margin-top: 2px; }

/* ── Empty state ── */
.ps-empty-state { background: #1a1d23; border: 1px dashed #2a2f3a; border-radius: 10px; padding: 40px 20px; text-align: center; }
.ps-empty-state p { color: #5a6170; font-size: 13px; margin: 0; }

/* ── Output box ── */
.ps-output-box { background: #13161c; border: 1px solid #2a2f3a; border-radius: 9px; padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; color: #e8eaed; line-height: 1.7; white-space: pre-wrap; word-break: break-word; min-height: 80px; }

/* ── Status badges ── */
.ps-badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 4px; font-size: 11.5px; font-weight: 600; }
.ps-badge.success { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.2); }
.ps-badge.error { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }

/* ── Buttons ── */
.stButton > button { border-radius: 8px !important; font-size: 13.5px !important; font-weight: 600 !important; transition: all 0.15s !important; }
.stButton > button[kind="primary"] { background: #3b82f6 !important; border: 1px solid #3b82f6 !important; color: white !important; }
.stButton > button[kind="primary"]:hover { background: #2563eb !important; border-color: #2563eb !important; }
.stButton > button[kind="secondary"] { background: #1a1d23 !important; border: 1px solid #2a2f3a !important; color: #e8eaed !important; }
.stButton > button[kind="secondary"]:hover { border-color: #353b47 !important; background: #22262e !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] { background: #1a1d23 !important; border: 1px dashed #2a2f3a !important; border-radius: 9px !important; }
[data-testid="stFileUploader"]:hover { border-color: #3b82f6 !important; }

/* ── Text area ── */
.stTextArea textarea { background: #1a1d23 !important; border: 1px solid #2a2f3a !important; border-radius: 8px !important; color: #e8eaed !important; font-size: 13.5px !important; }
.stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { color: #3b82f6 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #2a2f3a !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #9aa0b2 !important; font-size: 13px !important; font-weight: 500 !important; border-bottom: 2px solid transparent !important; padding: 8px 16px !important; }
.stTabs [aria-selected="true"] { color: #e8eaed !important; border-bottom-color: #3b82f6 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] { background: #1a1d23 !important; border: 1px solid #2a2f3a !important; border-radius: 9px !important; padding: 14px 16px !important; }
[data-testid="stMetricLabel"] { color: #9aa0b2 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #e8eaed !important; font-size: 22px !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; border-left-width: 3px !important; }

/* ── Divider ── */
hr { border-color: #2a2f3a !important; }

/* ── Code blocks ── */
.stCode { background: #13161c !important; border: 1px solid #2a2f3a !important; border-radius: 8px !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: #1a1d23 !important; border: 1px solid #2a2f3a !important; border-radius: 8px !important; color: #e8eaed !important; font-size: 13.5px !important; font-weight: 500 !important; }
.streamlit-expanderContent { border: 1px solid #2a2f3a !important; border-top: none !important; background: #161920 !important; }

/* ── Sidebar version footer ── */
.ps-sidebar-footer { padding: 12px 16px; font-size: 10.5px; color: #5a6170; border-top: 1px solid #2a2f3a; margin-top: 16px; }

/* ── Column labels ── */
.ps-col-label { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; color: #5a6170; text-transform: uppercase; padding-bottom: 10px; border-bottom: 1px solid #2a2f3a; margin-bottom: 16px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Helper functions ──────────────────────────────────────────────────────────
PAGES = ["Overview", "Encode", "Decode", "Analysis", "Documentation"]


def page_header(title: str, description: str) -> None:
    st.markdown(
        f'<div class="ps-page-header"><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="ps-section-label">{text}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="ps-metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="ps-metric-card">'
        f'<div class="ps-metric-label">{label}</div>'
        f'<div class="ps-metric-value">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )


def metrics_row(cards: list[tuple]) -> None:
    """Render a row of metric cards. Each tuple: (label, value, sub)."""
    inner = "".join(metric_card(*c) for c in cards)
    st.markdown(f'<div class="ps-metric-row">{inner}</div>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.markdown(
        f'<div class="ps-empty-state"><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def back_button() -> None:
    if st.button("← Back", key=f"back_{st.session_state.get('page','')}", use_container_width=False):
        st.session_state["page"] = "Overview"
        st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
<div class="ps-sidebar-brand">
  <div class="ps-logo">PS</div>
  <div>
    <div class="ps-brand-name">PhaseShift</div>
    <div class="ps-brand-sub">Frequency-Domain Steganography</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"

    current_index = PAGES.index(st.session_state["page"]) if st.session_state["page"] in PAGES else 0

    nav = st.radio(
        "Navigation",
        PAGES,
        index=current_index,
        label_visibility="collapsed",
    )
    if nav != st.session_state["page"]:
        st.session_state["page"] = nav
        st.rerun()

    st.markdown(
        '<div class="ps-sidebar-footer">v1.0.0 · Python · NumPy · Streamlit</div>',
        unsafe_allow_html=True,
    )

# ── Handle programmatic navigation ───────────────────────────────────────────
if "requested_page" in st.session_state:
    st.session_state["page"] = st.session_state.pop("requested_page")

# ── Page routing ──────────────────────────────────────────────────────────────
current_page = st.session_state.get("page", "Overview")


# ── Page: Overview ────────────────────────────────────────────────────────────
def render_overview() -> None:
    st.markdown(
        """
<div style="display:flex; flex-direction:column; align-items:center;
            text-align:center; padding: 60px 20px 32px;">

  <!-- Eyebrow label -->
  <div style="font-size:11px; font-weight:700; letter-spacing:0.2em; color:#3b82f6;
              text-transform:uppercase; margin-bottom:28px;">
    Frequency-Domain Steganography
  </div>

  <!-- Brand logo title -->
  <div style="display:inline-flex; align-items:center; gap:16px; margin-bottom:36px;">
    <div style="width:52px; height:52px; background:#3b82f6; border-radius:12px;
                display:flex; align-items:center; justify-content:center;
                font-size:22px; font-weight:900; color:white; letter-spacing:-1px;
                flex-shrink:0;">
      PS
    </div>
    <div style="font-size:68px; font-weight:800; letter-spacing:-3px; line-height:1;
                background: linear-gradient(90deg, #e8eaed 0%, #3b82f6 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text;">
      PhaseShift
    </div>
  </div>

  <!-- Description box -->
  <div style="width:100%; max-width:780px;
              background: linear-gradient(135deg, #1a1d23 0%, #1e2535 100%);
              border:1px solid #2e3d57; border-radius:12px;
              padding:28px 36px; text-align:left;
              box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);">
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:14px;">
      <div style="width:6px; height:6px; border-radius:50%; background:#3b82f6;"></div>
      <span style="font-size:11px; font-weight:700; letter-spacing:0.1em;
                   color:#3b82f6; text-transform:uppercase;">About this project</span>
    </div>
    <p style="font-size:15.5px; color:#9aa0b2; line-height:1.9; margin:0;">
      PhaseShift embeds secret text into images using the
      <strong style="color:#e8eaed;">2D Fast Fourier Transform</strong>.
      Rather than editing pixels directly, it works in the frequency domain —
      subtly shifting selected coefficients to encode a hidden bit stream.
      The result is a lossless PNG that looks identical to the original,
      but carries a payload recoverable only with the correct embedding strength
      and verified by a <strong style="color:#9aa0b2;">SHA-256 checksum</strong>.
    </p>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        if st.button("Encode", use_container_width=True):
            st.session_state["page"] = "Encode"
            st.rerun()
    with c2:
        if st.button("Decode", use_container_width=True):
            st.session_state["page"] = "Decode"
            st.rerun()
    with c3:
        if st.button("Analysis", use_container_width=True):
            st.session_state["page"] = "Analysis"
            st.rerun()
    with c4:
        if st.button("Documentation", use_container_width=True):
            st.session_state["page"] = "Documentation"
            st.rerun()


# ── Page: Encode ─────────────────────────────────────────────────────────────
def render_encode() -> None:
    back_button()
    page_header("Encode", "Hide a secret message inside a cover image using frequency-domain embedding.")

    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown('<div class="ps-col-label">Input</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Cover Image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="encode_upload",
        )

        preview_array: np.ndarray | None = None
        if uploaded is not None:
            try:
                preview_array = load_grayscale_image(uploaded.getvalue())
                h, w = preview_array.shape
                st.image(preview_array, caption=f"{w}×{h}px · Grayscale", width="stretch")
            except PhaseShiftError as exc:
                st.error(str(exc))
                preview_array = None

        message = st.text_area(
            "Secret Message",
            placeholder="Enter the message to hide…",
            height=120,
            max_chars=20000,
            key="encode_message",
        )
        if message:
            byte_count = len(message.encode("utf-8"))
            st.caption(f"{len(message)} chars · {byte_count} bytes")

        strength = st.slider(
            "Embedding Strength",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="encode_strength",
            help="Higher values improve recovery stability but may increase distortion. The decoder must use the same value.",
        )

        if preview_array is not None:
            h, w = preview_array.shape
            cap = max_payload_bytes(h, w)
            st.caption(f"Capacity: ~{cap:,} bytes available")

        run_encode = st.button(
            "⬡ Encode & Analyze",
            type="primary",
            use_container_width=True,
            key="btn_encode",
        )

    with col_out:
        st.markdown('<div class="ps-col-label">Output</div>', unsafe_allow_html=True)

        # Handle encode button press
        if run_encode:
            if uploaded is None:
                st.error("Upload a cover image first.")
            elif not message:
                st.error("Enter a non-empty secret message.")
            else:
                try:
                    with st.spinner("Running FFT, embedding payload, verifying recovery…"):
                        result = encode_message(uploaded.getvalue(), message, strength)
                    st.session_state["encode_result"] = result
                except PhaseShiftError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

        result = st.session_state.get("encode_result")

        if result is None:
            empty_state("Encode a message to see the stego image here.")
        else:
            m = result.metrics

            # Stego image preview
            st.image(result.stego, caption="Stego image (PNG)", width="stretch")

            # Metrics row
            psnr_val = "∞ dB" if math.isinf(m["psnr_db"]) else f"{m['psnr_db']:.2f} dB"
            metrics_row([
                ("Payload", f"{m['payload_bytes']:,}", "bytes"),
                ("PSNR", psnr_val, "quality"),
                ("Changed Px", f"{m['changed_pixel_percent']:.1f}%", "vs cover"),
                ("Capacity", f"{m['capacity_percent']:.1f}%", "used"),
            ])

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Download button
            st.download_button(
                "⬇ Download Stego PNG",
                data=result.stego_png,
                file_name=f"phaseshift_stego_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                type="primary",
                use_container_width=True,
            )

            # Status banner
            if result.checksum_ok and m.get("exact_recovery"):
                st.success("Checksum verified — message recovered successfully from the PNG.")
            else:
                st.error("Recovery check failed. Try a larger image or different strength.")

            # Tabs
            tab_vis, tab_met, tab_rec = st.tabs(["Visuals", "Metrics", "Recovery"])

            with tab_vis:
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(image_figure(result.cover, "Cover image"), use_container_width=True)
                    st.pyplot(
                        spectrum_figure(result.original_spectrum, "Cover spectrum"),
                        use_container_width=True,
                    )
                with c2:
                    st.pyplot(image_figure(result.stego, "Stego image"), use_container_width=True)
                    st.pyplot(
                        spectrum_figure(result.stego_spectrum, "Stego spectrum"),
                        use_container_width=True,
                    )
                st.pyplot(difference_figure(result.cover, result.stego), use_container_width=True)
                st.caption("Pixel differences are amplified for visibility.")

            with tab_met:
                table_rows = [
                    ("Image dimensions", f"{m['width']} × {m['height']}"),
                    ("Total pixels", f"{m['pixels']:,}"),
                    ("Message bytes (UTF-8)", f"{m['payload_bytes']:,}"),
                    ("Packet bytes (header + checksum)", f"{m['packet_bytes']:,}"),
                    ("Embedded bits", f"{m['bits_embedded']:,}"),
                    ("Coefficient capacity", f"{m['capacity_bits']:,} bits"),
                    ("Capacity used", f"{m['capacity_percent']:.2f}%"),
                    ("Embedding strength", f"{m['strength']:.2f}"),
                    ("Quantization step (δ)", f"{m['quantization_step']:.2f}"),
                    ("MSE", f"{m['mse']:.4f}"),
                    ("PSNR", "∞" if math.isinf(m["psnr_db"]) else f"{m['psnr_db']:.3f} dB"),
                    ("Changed pixels", f"{m['changed_pixel_percent']:.2f}%"),
                    ("Checksum", "PASS" if result.checksum_ok else "FAIL"),
                    ("Exact recovery", "Yes" if m.get("exact_recovery") else "No"),
                ]
                st.table({"Metric": [r[0] for r in table_rows], "Value": [r[1] for r in table_rows]})

            with tab_rec:
                st.markdown("**Recovered message**")
                st.code(result.recovered_message, language="text")
                chk_badge = (
                    '<span class="ps-badge success">✓ Checksum Verified</span>'
                    if result.checksum_ok
                    else '<span class="ps-badge error">✗ Checksum Failed</span>'
                )
                match_badge = (
                    '<span class="ps-badge success">✓ Exact Match</span>'
                    if m.get("exact_recovery")
                    else '<span class="ps-badge error">✗ Mismatch</span>'
                )
                st.markdown(
                    f"<div style='display:flex;gap:8px;margin-top:8px'>{chk_badge}{match_badge}</div>",
                    unsafe_allow_html=True,
                )


# ── Page: Decode ──────────────────────────────────────────────────────────────
def render_decode() -> None:
    back_button()
    page_header("Decode", "Upload a PhaseShift stego image to extract the hidden message.")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="ps-col-label">Upload</div>', unsafe_allow_html=True)

        decode_file = st.file_uploader(
            "Stego Image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="decode_upload",
        )

        if decode_file is not None:
            try:
                prev = load_grayscale_image(decode_file.getvalue())
                st.image(prev, caption="Uploaded stego image", width="stretch")
            except PhaseShiftError as exc:
                st.error(str(exc))

        decode_strength = st.slider(
            "Embedding Strength",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="decode_strength",
        )
        st.caption("Use the same strength value that was used during encoding.")

        run_decode = st.button(
            "⬡ Extract Message",
            type="primary",
            use_container_width=True,
            key="btn_decode",
        )

    with col_out:
        st.markdown('<div class="ps-col-label">Result</div>', unsafe_allow_html=True)

        if run_decode:
            if decode_file is None:
                st.error("Upload a stego image first.")
            else:
                try:
                    with st.spinner("Reading frequency coefficients and validating checksum…"):
                        decoded = extract_message(decode_file.getvalue(), decode_strength)
                    st.session_state["decode_result"] = decoded
                except PhaseShiftError as exc:
                    st.error(str(exc))
                    st.session_state.pop("decode_result", None)
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    st.session_state.pop("decode_result", None)

        decoded = st.session_state.get("decode_result")

        if decoded is None:
            empty_state("Extract a message to see results here.")
        else:
            # Metrics
            h, w = decoded.image_shape
            chk_badge_cls = "success" if decoded.checksum_ok else "error"
            chk_badge_txt = "✓ Verified" if decoded.checksum_ok else "✗ Failed"

            metrics_row([
                ("Payload", f"{decoded.payload_bytes:,}", "bytes"),
                ("Image", f"{w}×{h}", "pixels"),
            ])

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="margin-bottom:12px"><span class="ps-badge {chk_badge_cls}">{chk_badge_txt}</span></div>',
                unsafe_allow_html=True,
            )

            # Message output
            section_label("Extracted Message")
            safe_msg = decoded.message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="ps-output-box">{safe_msg}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Select all text above to copy.")

            if not decoded.checksum_ok:
                st.warning(
                    "Checksum failed — the recovered content may be corrupted. "
                    "Verify the embedding strength matches the one used during encoding, "
                    "and that the image has not been edited or re-compressed."
                )


# ── Page: Analysis ────────────────────────────────────────────────────────────
def render_analysis() -> None:
    back_button()
    page_header(
        "Analysis",
        "Frequency-domain comparison between cover and stego images.",
    )

    result = st.session_state.get("encode_result")

    if result is None:
        empty_state("No encoded result available. Run an encode first to see analysis.")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("Go to Encode", type="primary"):
            st.session_state["page"] = "Encode"
            st.rerun()
        return

    m = result.metrics

    # Metrics row
    psnr_val = "∞ dB" if math.isinf(m["psnr_db"]) else f"{m['psnr_db']:.2f} dB"
    metrics_row([
        ("MSE", f"{m['mse']:.4f}", "mean squared error"),
        ("PSNR", psnr_val, "signal-to-noise ratio"),
        ("Changed Pixels", f"{m['changed_pixel_percent']:.2f}%", "vs original"),
        ("Capacity Used", f"{m['capacity_percent']:.2f}%", f"{m['bits_embedded']:,} bits"),
    ])

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Image Comparison
    section_label("Image Comparison")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.pyplot(image_figure(result.cover, "Original Cover Image"), use_container_width=True)
    with c2:
        st.pyplot(image_figure(result.stego, "Stego Image"), use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Frequency Spectrum
    section_label("Frequency Spectrum")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.pyplot(
            spectrum_figure(result.original_spectrum, "Cover · Shifted FFT Spectrum"),
            use_container_width=True,
        )
    with c2:
        st.pyplot(
            spectrum_figure(result.stego_spectrum, "Stego · Shifted FFT Spectrum"),
            use_container_width=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Pixel Difference
    section_label("Pixel Difference")
    st.pyplot(difference_figure(result.cover, result.stego), use_container_width=True)
    st.caption(
        "The difference map amplifies pixel changes for visibility. "
        "Brightness here does not correspond to perceived visual distortion."
    )


# ── Page: Documentation ───────────────────────────────────────────────────────
def render_documentation() -> None:
    back_button()
    page_header("Documentation", "How PhaseShift works — technical reference and usage guide.")

    with st.expander("What is PhaseShift?", expanded=True):
        st.markdown(
            """
PhaseShift is an **educational prototype** for frequency-domain image steganography.
It embeds a UTF-8 text message into an image's Fourier frequency coefficients, producing
a stego image that is visually indistinguishable from the original.

> **Important:** This is an academic demonstration. It does not encrypt messages and should
> not be used to protect sensitive information.
"""
        )

    with st.expander("How Frequency-Domain Steganography Works"):
        st.markdown(
            """
The **2D Fast Fourier Transform (FFT)** decomposes an image from pixel space into a
representation of spatial frequencies. Low-frequency components capture broad structure;
high-frequency components capture fine edges and texture.

By slightly modifying the *amplitude* of selected frequency coefficients, a bit pattern
can be embedded. Since the changes are distributed across the frequency spectrum and
kept below the threshold of human visual perception, the stego image looks identical
to the original.
"""
        )

    with st.expander("The 2D FFT Process"):
        st.markdown("The encoding pipeline applied to each image:")
        st.code(
            "1. Load image → convert to grayscale (uint8 numpy array)\n"
            "2. Cast to float64 → apply numpy.fft.fft2()\n"
            "3. Apply numpy.fft.fftshift() → DC at center\n"
            "4. Select coefficient positions (avoid DC / near-DC region)\n"
            "5. Modify real parts using QIM\n"
            "6. ifftshift() → ifft2() → clip & round → save as PNG",
            language="text",
        )

    with st.expander("Message Encoding — Packet Structure"):
        st.markdown("Each message is wrapped in a binary packet before embedding:")
        st.code(
            "[ MAGIC (4 bytes) ]  PSH1\n"
            "[ LENGTH (4 bytes) ]  Big-endian UTF-8 payload length\n"
            "[ PAYLOAD (N bytes) ] UTF-8 encoded message\n"
            "[ CHECKSUM (4 bytes)] First 4 bytes of SHA-256(payload)",
            language="text",
        )
        st.markdown(
            """
Bits are embedded using **Quantization-Index Modulation (QIM)**:
each coefficient's real part is moved to one of two interleaved quantization lattices,
where the lattice index encodes a 0 or 1 bit.
Conjugate partner coefficients are updated simultaneously to maintain Hermitian symmetry
(required for real-valued inverse transforms).
"""
        )

    with st.expander("Message Decoding"):
        st.markdown(
            """
1. Load the stego image → compute FFT → fftshift
2. Read the same coefficient positions used during encoding
3. Quantize each real part with the same step δ — read the parity (0 or 1)
4. Reassemble bytes → verify magic header `PSH1`
5. Read payload length → extract payload → verify SHA-256 checksum
6. Decode UTF-8 bytes → return message string
"""
        )

    with st.expander("Limitations"):
        st.markdown(
            """
- **PNG required for lossless round-trip.** JPEG recompression destroys embedded data.
- **No encryption.** The message is stored as plain UTF-8 — anyone with PhaseShift can extract it.
- **Academic only.** Not audited for security; do not use for sensitive data.
- **Sensitivity.** Any editing, resizing, cropping, or recompression will likely corrupt the payload.
- **Fixed coefficient selection.** The embedding positions are deterministic and public.
"""
        )
        st.warning("Not suitable for production use. Academic demonstration only.")

    with st.expander("Supported Formats"):
        st.markdown(
            """
| Format | Recommended? | Notes |
|--------|-------------|-------|
| **PNG** | ✅ Yes | Lossless — required for reliable encode/decode |
| JPEG | ⚠️ Input only | Lossy — safe as cover input; **do not use as stego output** |
| BMP | ✅ OK | Lossless but large; PNG preferred |
| TIFF | ✅ OK | Lossless variants work; PNG preferred |
"""
        )


# ── Router ────────────────────────────────────────────────────────────────────
if current_page == "Overview":
    render_overview()
elif current_page == "Encode":
    render_encode()
elif current_page == "Decode":
    render_decode()
elif current_page == "Analysis":
    render_analysis()
elif current_page == "Documentation":
    render_documentation()
else:
    render_overview()
