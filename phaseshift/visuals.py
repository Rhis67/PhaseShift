"""Plotting helpers for the PhaseShift dashboard."""

import io

import matplotlib.pyplot as plt
import numpy as np


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def image_figure(array, title: str):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0e1628")
    ax.set_facecolor("#0e1628")
    ax.imshow(array, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, color="#f1f5f9", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def spectrum_figure(spectrum, title: str):
    magnitude = np.log1p(np.abs(spectrum))
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0e1628")
    ax.set_facecolor("#0e1628")
    ax.imshow(magnitude, cmap="magma")
    ax.set_title(title, color="#f1f5f9", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def difference_figure(cover, stego):
    difference = np.abs(cover.astype(float) - stego.astype(float))
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0e1628")
    ax.set_facecolor("#0e1628")
    ax.imshow(difference, cmap="inferno")
    ax.set_title("Absolute pixel difference (amplified)", color="#f1f5f9", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig
