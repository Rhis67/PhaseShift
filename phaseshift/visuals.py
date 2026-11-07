"""Plotting helpers for the PhaseShift dashboard."""

import io

import matplotlib.pyplot as plt
import numpy as np

BG = "#111318"
TEXT = "#e8eaed"


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def image_figure(array, title: str):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(array, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, color=TEXT, pad=12, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def spectrum_figure(spectrum, title: str):
    magnitude = np.log1p(np.abs(spectrum))
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(magnitude, cmap="magma")
    ax.set_title(title, color=TEXT, pad=12, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def difference_figure(cover, stego):
    difference = np.abs(cover.astype(float) - stego.astype(float))
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(difference, cmap="inferno")
    ax.set_title("Absolute pixel difference (amplified)", color=TEXT, pad=12, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig
