"""Plotly theme (LifeHistories colours on a quiet white layout), as in dental-3d-vae."""

import plotly.graph_objects as go
import plotly.io as pio

from biomedical_qa_peft.viz.colours import LIFE_HISTORIES

# Series colours in order, chosen with a colour-vision check: any first 2 or 3 stay
# distinguishable for colour-blind readers; from the 4th on, charts also label lines directly.
SERIES = tuple(LIFE_HISTORIES[i] for i in (6, 1, 5, 0, 2, 4, 3))
# Magnitude: light sand -> teal -> dark blue (lightness falls steadily).
SEQUENTIAL = [[0.0, LIFE_HISTORIES[3]], [0.55, LIFE_HISTORIES[5]], [1.0, LIFE_HISTORIES[6]]]
FONT = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"

# Display names in display order; each method keeps its SERIES colour in every figure.
METHODS = {
    "baseline": "Baseline",
    "prompt_engineering": "Prompt engineering",
    "prefix_tuning": "Prefix tuning",
    "lora": "LoRA",
}
METHOD_COLOURS = dict(zip(METHODS, SERIES))


def rgba(colour: str, alpha: float) -> str:
    """A palette hex colour as a CSS rgba() string with the given opacity."""
    red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def use_theme() -> None:
    """Make LifeHistories and a clean layout the default for new figures.

    Left-aligned titles with room for a subtitle, faint sand gridlines, no axis lines.
    """
    grid = rgba(LIFE_HISTORIES[3], 0.45)
    axis = dict(gridcolor=grid, zeroline=False, showline=False, ticks="", title_standoff=8)
    pio.templates["life_histories"] = go.layout.Template(
        layout=dict(
            colorway=list(SERIES),
            colorscale=dict(sequential=SEQUENTIAL),
            font=dict(family=FONT, size=13),
            title=dict(x=0, xanchor="left", font=dict(size=19)),
            xaxis=axis, yaxis=axis,
            legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", title_text=""),
            hoverlabel=dict(font=dict(family=FONT)),
            margin=dict(l=70, r=40, t=110, b=60),
        ),
        data=dict(heatmap=[go.Heatmap(colorscale=SEQUENTIAL)]),
    )
    pio.templates.default = "plotly_white+life_histories"
