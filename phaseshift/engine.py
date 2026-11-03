"""Core frequency-domain image steganography implementation.

This is an educational prototype, not a cryptographic or production-secure
steganography system. Use PNG for stego output; lossy recompression can damage
embedded data.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps

MAGIC = b"PSH1"
HEADER_BYTES = 8  # 4-byte magic + 4-byte UTF-8 payload length
CHECKSUM_BYTES = 4
MIN_IMAGE_DIMENSION = 32


class PhaseShiftError(ValueError):
    """Expected user-facing error from the PhaseShift engine."""


@dataclass
class EncodeResult:
    cover: np.ndarray
    stego: np.ndarray
    stego_png: bytes
    original_spectrum: np.ndarray
    stego_spectrum: np.ndarray
    recovered_message: str
    checksum_ok: bool
    metrics: dict[str, Any]


@dataclass
class DecodeResult:
    message: str
    checksum_ok: bool
    payload_bytes: int
    image_shape: tuple[int, int]


def load_grayscale_image(source: Any) -> np.ndarray:
    """Load a file path, bytes object, BytesIO, or PIL image as uint8 grayscale."""
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(source)
        image = ImageOps.exif_transpose(image).convert("L")
        array = np.asarray(image, dtype=np.uint8)
    except Exception as exc:
        raise PhaseShiftError(f"Could not read image: {exc}") from exc

    if array.ndim != 2:
        raise PhaseShiftError("Image could not be converted to a grayscale matrix.")
    if min(array.shape) < MIN_IMAGE_DIMENSION:
        raise PhaseShiftError(
            f"Image is too small ({array.shape[1]} × {array.shape[0]}). "
            f"Use an image at least {MIN_IMAGE_DIMENSION} × {MIN_IMAGE_DIMENSION} pixels."
        )
    return array


def array_to_png_bytes(array: np.ndarray) -> bytes:
    """Serialize a grayscale array to lossless PNG bytes."""
    safe = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(safe, mode="L").save(output, format="PNG")
    return output.getvalue()


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) % 8:
        raise PhaseShiftError("Internal bit stream is not byte-aligned.")
    return np.packbits(bits).tobytes()


def _make_packet(message: str) -> bytes:
    payload = message.encode("utf-8")
    if len(payload) > 0xFFFFFFFF:
        raise PhaseShiftError("Message is too large.")
    checksum = hashlib.sha256(payload).digest()[:CHECKSUM_BYTES]
    return MAGIC + len(payload).to_bytes(4, "big") + payload + checksum


def _candidate_positions(height: int, width: int):
    """Yield a deterministic set of coefficients from one half-spectrum.

    The DC/near-DC region is avoided. Only one side is enumerated, so a
    coefficient and its conjugate partner are not both independently selected.
    """
    row_mid, col_mid = height // 2, width // 2
    for row in range(1, height - 1):
        for col in range(1, max(2, width // 2)):
            if abs(row - row_mid) <= 2 and abs(col - col_mid) <= 2:
                continue
            yield row, col


def capacity_bits(height: int, width: int) -> int:
    """Return approximate raw usable coefficient count in bits."""
    return sum(1 for _ in _candidate_positions(height, width))


def max_payload_bytes(height: int, width: int) -> int:
    """Approximate maximum message payload after packet overhead."""
    overhead = HEADER_BYTES + CHECKSUM_BYTES
    return max(0, (capacity_bits(height, width) // 8) - overhead)


def _partner(row: int, col: int, height: int, width: int) -> tuple[int, int]:
    """Find the conjugate partner in fftshifted coordinates."""
    unshifted_row = (row - height // 2) % height
    unshifted_col = (col - width // 2) % width
    partner_row = (-unshifted_row) % height
    partner_col = (-unshifted_col) % width
    return (
        (partner_row + height // 2) % height,
        (partner_col + width // 2) % width,
    )


def _read_bits(spectrum: np.ndarray, count: int, delta: float) -> np.ndarray:
    height, width = spectrum.shape
    result = []
    for row, col in _candidate_positions(height, width):
        if len(result) >= count:
            break
        quantized_index = int(np.rint(float(spectrum[row, col].real) / delta))
        result.append(quantized_index % 2)
    if len(result) < count:
        raise PhaseShiftError("Image does not have enough usable frequency coefficients.")
    return np.asarray(result, dtype=np.uint8)


def _embed_packet(
    image: np.ndarray, packet: bytes, strength: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    height, width = image.shape
    bits = _bytes_to_bits(packet)
    available = capacity_bits(height, width)
    if len(bits) > available:
        raise PhaseShiftError(
            f"Message needs {len(bits)} bits, but this image has about {available} usable bits. "
            f"Try a larger image or a shorter message. Approximate maximum payload: "
            f"{max_payload_bytes(height, width)} bytes."
        )

    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    original_spectrum = spectrum.copy()
    # Coefficients scale with image area. The strength controls quantization spacing.
    delta = max(2.0, float(strength) * image.size)

    for bit, (row, col) in zip(bits, _candidate_positions(height, width)):
        value = spectrum[row, col]
        partner_row, partner_col = _partner(row, col, height, width)
        lattice_index = int(np.rint(float(value.real) / (2.0 * delta)))
        target_real = (2 * lattice_index + int(bit)) * delta
        spectrum[row, col] = target_real + 1j * value.imag
        spectrum[partner_row, partner_col] = target_real - 1j * value.imag

    reconstructed = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
    stego = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)
    return stego, original_spectrum, spectrum, delta


def extract_message(image: Any, strength: float = 0.05) -> DecodeResult:
    """Extract a message from a PhaseShift image and check its checksum."""
    if not 0.01 <= float(strength) <= 0.20:
        raise PhaseShiftError("Embedding strength must be between 0.01 and 0.20.")

    array = load_grayscale_image(image)
    spectrum = np.fft.fftshift(np.fft.fft2(array.astype(np.float64)))
    delta = max(2.0, float(strength) * array.size)

    header_bits = _read_bits(spectrum, HEADER_BYTES * 8, delta)
    header = _bits_to_bytes(header_bits)
    if header[:4] != MAGIC:
        raise PhaseShiftError(
            "No valid PhaseShift header was found. This may not be a PhaseShift image, "
            "or the embedding strength may be incorrect."
        )

    payload_length = int.from_bytes(header[4:8], "big")
    total_bytes = HEADER_BYTES + payload_length + CHECKSUM_BYTES
    total_bits = total_bytes * 8
    if total_bits > capacity_bits(*array.shape):
        raise PhaseShiftError("The embedded length exceeds the usable image capacity.")

    packet = _bits_to_bytes(_read_bits(spectrum, total_bits, delta))
    payload_start = HEADER_BYTES
    payload_end = payload_start + payload_length
    payload = packet[payload_start:payload_end]
    stored_checksum = packet[payload_end:payload_end + CHECKSUM_BYTES]
    expected_checksum = hashlib.sha256(payload).digest()[:CHECKSUM_BYTES]
    checksum_ok = stored_checksum == expected_checksum

    try:
        message = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PhaseShiftError(
            "The recovered bytes are not valid UTF-8. Check the image and embedding strength."
        ) from exc

    return DecodeResult(
        message=message,
        checksum_ok=checksum_ok,
        payload_bytes=len(payload),
        image_shape=array.shape,
    )


def encode_message(
    image: Any, message: str, strength: float = 0.05
) -> EncodeResult:
    """Embed a UTF-8 message, save as PNG, then verify by decoding the saved PNG."""
    if not isinstance(message, str) or not message:
        raise PhaseShiftError("Enter a non-empty secret message.")
    if not 0.01 <= float(strength) <= 0.20:
        raise PhaseShiftError("Embedding strength must be between 0.01 and 0.20.")

    cover = load_grayscale_image(image)
    packet = _make_packet(message)
    stego, original_spectrum, embedded_spectrum, delta = _embed_packet(cover, packet, strength)
    png_bytes = array_to_png_bytes(stego)

    # Verify the actual serialized PNG artifact.
    decoded = extract_message(png_bytes, strength)
    stego_reopened = load_grayscale_image(png_bytes)

    cover_float = cover.astype(np.float64)
    stego_float = stego_reopened.astype(np.float64)
    mse = float(np.mean((cover_float - stego_float) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
    changed_fraction = float(np.mean(cover != stego_reopened))
    payload_bytes = len(message.encode("utf-8"))
    used_bits = len(packet) * 8
    total_bits = capacity_bits(*cover.shape)

    metrics = {
        "width": int(cover.shape[1]),
        "height": int(cover.shape[0]),
        "pixels": int(cover.size),
        "payload_bytes": payload_bytes,
        "packet_bytes": len(packet),
        "bits_embedded": used_bits,
        "capacity_bits": total_bits,
        "capacity_percent": float(100.0 * used_bits / total_bits) if total_bits else 0.0,
        "strength": float(strength),
        "quantization_step": float(delta),
        "mse": mse,
        "psnr_db": psnr,
        "changed_pixel_percent": changed_fraction * 100.0,
        "checksum_ok": bool(decoded.checksum_ok),
        "exact_recovery": bool(decoded.message == message),
    }

    return EncodeResult(
        cover=cover,
        stego=stego_reopened,
        stego_png=png_bytes,
        original_spectrum=original_spectrum,
        stego_spectrum=np.fft.fftshift(np.fft.fft2(stego_reopened.astype(np.float64))),
        recovered_message=decoded.message,
        checksum_ok=decoded.checksum_ok,
        metrics=metrics,
    )


def image_quality_metrics(cover: Any, stego: Any) -> dict[str, float]:
    """Calculate MSE, PSNR, and changed-pixel percentage for same-sized images."""
    a = load_grayscale_image(cover).astype(np.float64)
    b = load_grayscale_image(stego).astype(np.float64)
    if a.shape != b.shape:
        raise PhaseShiftError("Images must have the same dimensions to compare quality.")
    mse = float(np.mean((a - b) ** 2))
    return {
        "mse": mse,
        "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse)),
        "changed_pixel_percent": float(100 * np.mean(a != b)),
    }
