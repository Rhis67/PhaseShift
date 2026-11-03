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

st.set_page_config(
    page_title="PhaseShift | Steganography Lab",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: radial-gradient(ellipse at top left, #182746 0%, #0b1220 48%, #080d18 100%); }
      [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #24324c; }
      .hero { padding: 1.35rem 1.5rem; border: 1px solid #2b3b59; border-radius: 20px;
        background: linear-gradient(120deg, rgba(30,48,82,.94), rgba(13,22,39,.90)); }
      .eyebrow { color: #8db5ff; text-transform: uppercase; letter-spacing: .16em;
        font-size: .75rem; font-weight: 700; }
      .muted { color: #a9b8d0; }
      .section { border: 1px solid #263550; border-radius: 16px; padding: 1rem 1.1rem;
        background: rgba(14, 24, 42, .7); }
      div[data-testid="stMetric"] { border: 1px solid #2b3b59; border-radius: 14px;
        padding: 12px 14px; background: rgba(18, 31, 53, .8); }
      div.stButton > button[kind="primary"] { border-radius: 10px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Frequency-domain image steganography</div>
      <h1 style="margin:.35rem 0 .25rem 0;">🌌 PhaseShift</h1>
      <p class="muted" style="margin:0;">Hide a message in an image's frequency coefficients, inspect the transformation,
      and verify recovery from the saved PNG.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🌌 PhaseShift")
    st.caption("Steganography Lab · Version 1.0")
    page = st.radio(
        "Workspace",
        ["Overview", "Encode message", "Decode message", "Algorithm & metrics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("#### Quick guide")
    st.caption("1. Upload a cover image")
    st.caption("2. Encode a short message")
    st.caption("3. Download the PNG")
    st.caption("4. Decode using the same strength")
    st.divider()
    st.caption("PNG is recommended. JPEG recompression can corrupt hidden data.")

if page == "Overview":
    st.markdown("### Your steganography workspace")
    st.write("PhaseShift demonstrates how an image can be represented in the frequency domain and how selected Fourier coefficients can carry a text payload.")
    cards = st.columns(3)
    with cards[0]:
        st.markdown('<div class="section"><h3>🖼️ Cover image</h3><p class="muted">Load and inspect a grayscale image before encoding.</p></div>', unsafe_allow_html=True)
    with cards[1]:
        st.markdown('<div class="section"><h3>〰️ Frequency lab</h3><p class="muted">Explore the shifted FFT spectrum and the reconstructed stego image.</p></div>', unsafe_allow_html=True)
    with cards[2]:
        st.markdown('<div class="section"><h3>🔎 Recovery check</h3><p class="muted">Extract the payload and validate its SHA-256 checksum.</p></div>', unsafe_allow_html=True)
    st.markdown("### Workflow")
    st.code("Cover image → Grayscale matrix → 2D FFT → Coefficient embedding → Inverse FFT → PNG → Extraction → Checksum", language="text")
    st.info("This is an educational demonstration. Hiding a message is not the same as encrypting it.")
    st.markdown("### Start here")
    if st.button("Open the encoding workspace", type="primary"):
        st.session_state["requested_page"] = "Encode message"
        st.rerun()

elif page == "Encode message":
    st.markdown("### 01 · Encode a message")
    st.write("Select an image, type a message, and generate a downloadable PNG.")
    col_input, col_method = st.columns([1.15, 0.85], gap="large")

    with col_input:
        uploaded = st.file_uploader("Cover image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="encode_upload")
        message = st.text_area("Secret message", placeholder="Enter the message to hide…", height=130, max_chars=20000)
        strength = st.slider(
            "Embedding strength",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="The decoder must use the same value. Larger values usually improve recovery stability but can increase distortion.",
        )
        if uploaded is not None:
            try:
                preview = load_grayscale_image(uploaded.getvalue())
                payload_max = max_payload_bytes(*preview.shape)
                st.caption(f"Image: {preview.shape[1]} × {preview.shape[0]} px · Approx. max payload: {payload_max:,} bytes")
            except PhaseShiftError as exc:
                st.error(str(exc))
        run_encode = st.button("✨ Encode & analyze", type="primary", use_container_width=True)

    with col_method:
        st.markdown('<div class="section"><h4>What the engine does</h4><ol><li>Converts the image to grayscale.</li><li>Computes a 2D FFT and shifts the spectrum.</li><li>Encodes UTF-8 bits in selected coefficient pairs.</li><li>Runs the inverse FFT and saves lossless PNG.</li><li>Reopens the PNG and attempts recovery.</li></ol></div>', unsafe_allow_html=True)
        st.warning("Do not use JPEG as the final stego output. Lossy compression can damage frequency coefficients.")

    if run_encode:
        if uploaded is None:
            st.error("Upload a cover image first.")
        elif not message:
            st.error("Enter a non-empty secret message.")
        else:
            try:
                with st.spinner("Running FFT, embedding payload, reconstructing image and verifying recovery…"):
                    result = encode_message(uploaded.getvalue(), message, strength)
                st.session_state["encode_result"] = result
                st.session_state["encode_filename"] = "phaseshift_stego.png"
            except PhaseShiftError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    result = st.session_state.get("encode_result")
    if result is not None:
        st.divider()
        st.markdown("### Results")
        metrics = result.metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Payload", f"{metrics['payload_bytes']:,} bytes")
        m2.metric("PSNR", "∞" if math.isinf(metrics["psnr_db"]) else f"{metrics['psnr_db']:.2f} dB")
        m3.metric("Pixels changed", f"{metrics['changed_pixel_percent']:.1f}%")
        m4.metric("Checksum", "PASS" if result.checksum_ok and metrics["exact_recovery"] else "FAIL")
        if result.checksum_ok and metrics["exact_recovery"]:
            st.success("The saved PNG successfully recovered the original message and passed checksum verification.")
        else:
            st.error("The recovery test did not pass. Try a different strength or a larger image.")

        tab_visual, tab_metrics, tab_recovery = st.tabs(["Visual comparison", "Technical metrics", "Recovered message"])
        with tab_visual:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(image_figure(result.cover, "Original cover"), use_container_width=True)
                st.pyplot(spectrum_figure(result.original_spectrum, "Cover · shifted FFT spectrum"), use_container_width=True)
            with c2:
                st.pyplot(image_figure(result.stego, "Stego image"), use_container_width=True)
                st.pyplot(spectrum_figure(result.stego_spectrum, "Stego · shifted FFT spectrum"), use_container_width=True)
            st.pyplot(difference_figure(result.cover, result.stego), use_container_width=True)
            st.caption("The difference view amplifies pixel changes. Its brightness is not the same as perceived visual distortion.")
        with tab_metrics:
            table_rows = [
                ("Image dimensions", f"{metrics['width']} × {metrics['height']}"),
                ("Total pixels", f"{metrics['pixels']:,}"),
                ("Message bytes (UTF-8)", f"{metrics['payload_bytes']:,}"),
                ("Packet bytes including header/checksum", f"{metrics['packet_bytes']:,}"),
                ("Embedded bits", f"{metrics['bits_embedded']:,}"),
                ("Approximate coefficient capacity", f"{metrics['capacity_bits']:,} bits"),
                ("Capacity used", f"{metrics['capacity_percent']:.2f}%"),
                ("Embedding strength", f"{metrics['strength']:.2f}"),
                ("Quantization step", f"{metrics['quantization_step']:.2f}"),
                ("Mean squared error (MSE)", f"{metrics['mse']:.4f}"),
                ("Peak signal-to-noise ratio (PSNR)", "∞" if math.isinf(metrics["psnr_db"]) else f"{metrics['psnr_db']:.3f} dB"),
                ("Changed pixels", f"{metrics['changed_pixel_percent']:.2f}%"),
            ]
            st.table({"Metric": [r[0] for r in table_rows], "Value": [r[1] for r in table_rows]})
            st.caption("Capacity is an approximate upper bound based on the selected coefficient region. Actual robustness depends on the image and strength.")
        with tab_recovery:
            st.markdown("**Recovered text**")
            st.code(result.recovered_message, language="text")
            st.write("Exact match:", "Yes" if metrics["exact_recovery"] else "No")
            st.write("Checksum valid:", "Yes" if result.checksum_ok else "No")

        download_col, info_col = st.columns([1, 2])
        with download_col:
            st.download_button(
                "⬇️ Download stego PNG",
                data=result.stego_png,
                file_name=f"phaseshift_stego_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                type="primary",
                use_container_width=True,
            )
        with info_col:
            st.caption("Keep the embedding strength value. You will need the same value in the Decode workspace.")

elif page == "Decode message":
    st.markdown("### 02 · Decode and verify")
    st.write("Upload the PNG generated by PhaseShift. Select the same embedding strength used during encoding.")
    decode_file = st.file_uploader("Stego image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="decode_upload")
    decode_strength = st.slider("Embedding strength used during encoding", 0.01, 0.20, 0.05, 0.01, key="decode_strength")
    if st.button("🔎 Extract message", type="primary", use_container_width=True):
        if decode_file is None:
            st.error("Upload a stego image first.")
        else:
            try:
                with st.spinner("Reading frequency coefficients and validating checksum…"):
                    decoded = extract_message(decode_file.getvalue(), decode_strength)
                st.session_state["decode_result"] = decoded
            except PhaseShiftError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    decoded = st.session_state.get("decode_result")
    if decoded is not None:
        st.divider()
        st.markdown("### Extraction result")
        st.metric("Recovered payload", f"{decoded.payload_bytes:,} bytes")
        st.code(decoded.message, language="text")
        if decoded.checksum_ok:
            st.success("Checksum verified. The recovered payload is internally consistent.")
        else:
            st.error("Checksum mismatch. The image may have been altered or the strength may be wrong.")
        st.caption(f"Image dimensions: {decoded.image_shape[1]} × {decoded.image_shape[0]} px")

else:
    st.markdown("### 03 · Algorithm & metrics")
    st.markdown("#### Processing pipeline")
    st.code(
        "Text → UTF-8 bytes → Header + length + checksum → Binary bits\n"
        "Image → Grayscale matrix → 2D FFT → fftshift\n"
        "Bits → Quantization-index modulation of coefficient pairs\n"
        "Modified spectrum → ifftshift → Inverse FFT → rounded grayscale PNG\n"
        "Saved PNG → FFT → bit extraction → UTF-8 message + checksum verification",
        language="text",
    )
    st.markdown("#### Core concepts")
    with st.expander("What is the 2D FFT?"):
        st.write("The two-dimensional Fast Fourier Transform represents an image as spatial-frequency coefficients. Low frequencies describe slowly changing image structure; higher frequencies represent finer detail and edges. `fftshift` moves the low-frequency center to the middle of the displayed spectrum.")
    with st.expander("Why modify conjugate coefficient pairs?"):
        st.write("For a real-valued image, Fourier coefficients have conjugate symmetry. Updating a coefficient and its conjugate partner consistently helps the inverse transform remain real-valued.")
    with st.expander("What is quantization-index modulation?"):
        st.write("A coefficient's real component is moved to one of two interleaved quantization lattices. The lattice choice represents a 0 or 1. The decoder quantizes the coefficient and reads the parity to estimate the bit.")
    with st.expander("What do MSE and PSNR mean?"):
        st.write("MSE is the average squared pixel difference between the cover and stego images. PSNR expresses the error on a logarithmic decibel scale. Higher PSNR generally indicates lower pixel error, but neither metric alone guarantees that the image looks identical.")
    with st.expander("What does the checksum prove?"):
        st.write("A truncated SHA-256 checksum detects many accidental extraction errors. It is not encryption, authentication, or proof that an image has not been tampered with.")
    st.warning("This implementation is for academic demonstration. It has not been independently audited and should not be used to protect sensitive messages.")
    st.markdown("#### Engineering notes")
    st.write("- Use PNG to avoid lossy compression during the encode/decode cycle.")
    st.write("- The message is encoded as UTF-8, so non-English text can be represented.")
    st.write("- The decoder needs the same embedding strength used during encoding.")
    st.write("- The algorithm is sensitive to resizing, cropping, editing, and recompression.")
    st.write("- Test multiple image textures and sizes before reporting experimental results.")

st.markdown("---")
st.caption("PhaseShift · Academic prototype · Python / NumPy / Pillow / Matplotlib / Streamlit")
