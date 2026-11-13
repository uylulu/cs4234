"""Illustrative vertex layouts for the two selected weighted-graph families.

Positions illustrate planted memberships, not an embedding or measured features.
Actual archive memberships and matching endpoints determine all plotted marks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/report_figures"


def export(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    rng = np.random.default_rng(4234)
    z = np.load(ROOT / "data/processed/synthetic/balanced_nested.npz")
    order = z["display_order"]
    labels = z["labels"][order]
    assert np.array_equal(np.bincount(labels), np.full(8, 16))
    fig, ax = plt.subplots(figsize=(10, 6.3))
    fig.subplots_adjust(top=0.81, bottom=0.17, left=0.025, right=0.975)
    fig.suptitle("Balanced nested: groups inside groups", fontsize=20, y=0.98, weight="bold")
    fig.text(
        0.5,
        0.90,
        "128 points  →  2 groups of 64  →  4 groups of 32  →  8 groups of 16",
        ha="center",
        fontsize=12,
    )
    palettes = [
        "#276b9e",
        "#24a4b5",
        "#5476c3",
        "#60a886",
        "#925498",
        "#c15b7a",
        "#aa7a3b",
        "#b39125",
    ]
    for top in range(2):
        center_x = -2.5 if top == 0 else 2.5
        ax.add_patch(
            FancyBboxPatch(
                (center_x - 1.95, -2.35),
                3.9,
                4.7,
                boxstyle="round,pad=0.08,rounding_size=0.18",
                facecolor="#eef4f8" if top == 0 else "#f8f0f5",
                edgecolor="#8b9dad",
                linewidth=1.3,
                zorder=0,
            )
        )
        ax.text(
            center_x,
            2.58,
            f"Large group {'A' if top == 0 else 'B'}: 64 points",
            ha="center",
            va="center",
            weight="bold",
        )
        for middle in range(2):
            cy = 1.15 if middle == 0 else -1.15
            ax.add_patch(
                FancyBboxPatch(
                    (center_x - 1.72, cy - 0.92),
                    3.44,
                    1.84,
                    boxstyle="round,pad=0.03,rounding_size=0.14",
                    facecolor="none",
                    edgecolor="#9ea9b3",
                    linestyle="--",
                    linewidth=1.0,
                )
            )
            for leaf in range(2):
                group = top * 4 + middle * 2 + leaf
                cx = center_x + (-0.84 if leaf == 0 else 0.84)
                ax.add_patch(
                    Ellipse(
                        (cx, cy),
                        1.35,
                        1.20,
                        facecolor=palettes[group],
                        edgecolor="none",
                        alpha=0.10,
                    )
                )
                indices = np.flatnonzero(labels == group)
                radius = np.sqrt(rng.uniform(0, 1, len(indices)))
                angle = rng.uniform(0, 2 * np.pi, len(indices))
                x = cx + 0.52 * radius * np.cos(angle)
                y = cy + 0.44 * radius * np.sin(angle)
                ax.scatter(x, y, s=26, c=palettes[group], edgecolors="white", linewidths=0.4)
                ax.text(cx, cy - 0.72, f"G{group + 1} · 16 points", ha="center", fontsize=10)
    ax.set_xlim(-4.7, 4.7)
    ax.set_ylim(-2.6, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(
        0.5,
        0.095,
        "Closer relatives in the hierarchy have higher similarity.",
        ha="center",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.045,
        "Illustrative positions; enclosing regions show planted groups, not physical distances.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    export(fig, "balanced_nested_points")

    z = np.load(ROOT / "data/processed/exploratory_cross_pairs/cross_pairs_5000.npz")
    order = z["display_order"]
    planted = z["planted_similarities"][np.ix_(order, order)]
    assert np.all(planted[np.arange(16), np.arange(16) + 16] == 1)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    left = np.c_[-2.25 + 1.05 * np.cos(angles), 1.22 * np.sin(angles)]
    right = np.c_[2.25 + 1.05 * np.cos(angles + 0.18), 1.22 * np.sin(angles + 0.18)]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.04, right=0.96)
    fig.suptitle(
        "Cross-pair stress: strong partners across two communities",
        fontsize=18,
        y=0.98,
        weight="bold",
    )
    fig.text(
        0.5,
        0.895,
        "32 points  •  2 groups of 16  •  16 designated cross-group pairs",
        ha="center",
        fontsize=12,
    )
    for points, color, name in [(left, "#247ba5", "Group A"), (right, "#8858a5", "Group B")]:
        edges = [[points[i], points[j]] for i in range(16) for j in range(i + 1, 16)]
        ax.add_collection(
            LineCollection(edges, colors=color, linewidths=0.65, alpha=0.13, zorder=0)
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            s=105,
            edgecolors="white",
            linewidth=1,
            zorder=3,
        )
        ax.text(
            points[:, 0].mean(), 1.62, f"{name}: 16 points", ha="center", weight="bold", fontsize=12
        )
    shown = [0, 4, 8, 12]
    for number, i in enumerate(shown, start=1):
        ax.plot(
            [left[i, 0], right[i, 0]],
            [left[i, 1], right[i, 1]],
            color="#d57520",
            linewidth=2.5,
            alpha=0.85,
            zorder=1,
        )
        for point in (left[i], right[i]):
            ax.scatter(*point, color="#d57520", edgecolors="white", linewidth=1, s=160, zorder=4)
            ax.text(
                *point, str(number), color="white", ha="center", va="center", fontsize=8.5, zorder=5
            )
    ax.set_xlim(-3.55, 3.55)
    ax.set_ylim(-1.6, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="#507f9e", lw=1.5, label="Within-group links: weight 0.6"),
        Line2D([0], [0], color="#d57520", lw=2.5, label="Designated cross-pairs: weight 1.0"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        frameon=False,
        ncol=2,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.085,
        "Four of the 16 strong cross-pairs are shown. Other cross-group links have weight 0.02 (hidden).",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.035,
        "Illustrative positions; weights shown are before noise. Each orange number labels one matched pair.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    export(fig, "cross_pair_points")
    print("Created two labelled vertex schematics (PNG and SVG)")


if __name__ == "__main__":
    main()
