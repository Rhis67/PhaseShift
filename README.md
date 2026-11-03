# PhaseShift: Frequency-Domain Image Steganography

A complete Python implementation and Streamlit dashboard for an academic demonstration of frequency-domain image steganography.

## Features

- Upload cover images and convert them to grayscale.
- Compute and display a shifted 2D FFT spectrum.
- Embed UTF-8 text into selected Fourier coefficient pairs using quantization-index modulation.
- Reconstruct and download a lossless PNG stego image.
- Extract a message from a saved stego image.
- Verify the recovered payload with a truncated SHA-256 checksum.
- Visualize cover, stego, spectrum, and amplified pixel-difference images.
- Report payload size, approximate capacity use, MSE, PSNR, and changed-pixel percentage.
- Automated engine tests, including Unicode round trips.

## Project structure

```text
PhaseShift_Full_Application/
├── app.py
├── requirements.txt
├── README.md
├── phaseshift/
│   ├── __init__.py
│   ├── engine.py
│   └── visuals.py
├── tests/
│   └── test_engine.py
└── samples/
```

## Requirements

- Python 3.10+
- Windows 10/11, macOS, or Linux
- Internet connection for initial package installation

## Run on Windows with VS Code

1. Extract the ZIP and open the `PhaseShift_Full_Application` folder in VS Code.
2. Open the integrated terminal (`Ctrl + `` `).
3. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Start the app:

```powershell
python -m streamlit run app.py
```

6. Open the local URL printed in the terminal, normally `http://localhost:8501`.

### If PowerShell blocks activation

You can run without activating the virtual environment:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Run the tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

## Basic workflow

1. Open **Encode message**.
2. Upload an image. For the first test, use a textured PNG or JPEG of at least 256 × 256 pixels.
3. Enter a short message and start with embedding strength `0.05`.
4. Click **Encode & analyze**.
5. Inspect the original image, FFT spectrum, stego image, difference view, and recovery status.
6. Download the stego PNG.
7. Open **Decode message**, upload that PNG, and use the same embedding strength.

## Implementation overview

1. Convert the cover image to an 8-bit grayscale matrix.
2. Apply `numpy.fft.fft2` and center the spectrum with `numpy.fft.fftshift`.
3. Convert the UTF-8 message to bytes and add a magic header, payload length, and checksum.
4. Encode bits by moving selected coefficient real components onto alternating quantization lattices.
5. Update conjugate coefficient pairs to preserve Fourier conjugate symmetry.
6. Apply the inverse transform, round to pixel values, and save PNG.
7. Reopen the saved PNG, extract the bit stream, decode UTF-8, and validate the checksum.

## Important limitations

- This is an educational prototype, not a production-grade or cryptographically secure system.
- The payload is hidden, not encrypted.
- Use PNG for the output. JPEG recompression, resizing, cropping, and edits can damage the payload.
- The embedding strength must match between encoding and decoding.
- Capacity is approximate and depends on image dimensions and the chosen coefficient region.
- A checksum detects many accidental errors but does not authenticate the sender or prevent tampering.
- PSNR and MSE are useful measurements but do not by themselves prove imperceptibility.

## Suggested viva explanation

- **Fitness / objective:** not applicable; this is not a genetic algorithm. The key design trade-off is payload capacity, visual distortion, and successful recovery.
- **FFT:** converts spatial image data into frequency coefficients.
- **fftshift:** places low frequencies near the center for inspection and coefficient selection.
- **Inverse FFT:** converts the modified spectrum back into an image.
- **Conjugate symmetry:** helps reconstruct a real-valued image from a modified spectrum.
- **Quantization-index modulation:** represents a bit by selecting one of two quantization lattices.
- **Checksum:** checks whether recovered payload bytes match the checksum stored with the message.
- **PSNR/MSE:** quantify pixel-level distortion between cover and stego images.

Be ready to explain the limitations and show actual test results. Do not claim robustness against JPEG compression unless you have measured it experimentally.
