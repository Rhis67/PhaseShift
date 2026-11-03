import io
import unittest

import numpy as np
from PIL import Image

from phaseshift.engine import (
    PhaseShiftError,
    capacity_bits,
    encode_message,
    extract_message,
    load_grayscale_image,
)


def make_test_png(width=256, height=256):
    # Deterministic textured image provides enough frequency variation for tests.
    y, x = np.mgrid[0:height, 0:width]
    image = (127 + 48 * np.sin(x / 7.0) + 42 * np.cos(y / 11.0) + 22 * np.sin((x + y) / 5.0))
    image = np.clip(image, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


class PhaseShiftEngineTests(unittest.TestCase):
    def test_encode_then_decode(self):
        source = make_test_png()
        message = "PhaseShift test: hello 🌌"
        result = encode_message(source, message, strength=0.05)
        self.assertEqual(result.recovered_message, message)
        self.assertTrue(result.checksum_ok)
        decoded = extract_message(result.stego_png, strength=0.05)
        self.assertEqual(decoded.message, message)
        self.assertTrue(decoded.checksum_ok)

    def test_unicode_round_trip(self):
        source = make_test_png()
        message = "नमस्ते Rhis"
        result = encode_message(source, message, strength=0.05)
        self.assertEqual(result.recovered_message, message)

    def test_too_small_image_rejected(self):
        image = np.zeros((16, 16), dtype=np.uint8)
        with self.assertRaises(PhaseShiftError):
            load_grayscale_image(Image.fromarray(image))

    def test_unencoded_image_rejected(self):
        source = make_test_png()
        with self.assertRaises(PhaseShiftError):
            extract_message(source, strength=0.05)

    def test_capacity_is_positive(self):
        self.assertGreater(capacity_bits(256, 256), 0)


if __name__ == "__main__":
    unittest.main()
