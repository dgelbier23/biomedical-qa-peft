"""Result figures for the four methods.

``runs`` arguments map a method key (see ``METHODS``) to its loaded ``results.json``, in
display order. Colours follow the method, so each method looks the same in every figure.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from biomedical_qa_peft.evaluation.metrics import mcnemar_p
from biomedical_qa_peft.viz.colours import LIFE_HISTORIES
from biomedical_qa_peft.viz.plotting import METHOD_COLOURS, METHODS, SEQUENTIAL, rgba

NLI_COLOURS = {"entailment": LIFE_HISTORIES[5], "neutral": LIFE_HISTORIES[3], "contradiction": LIFE_HISTORIES[1]}
# Multi-panel charts: legend centred below the panels, clear of the panel titles.
PANEL_LAYOUT = dict(legend=dict(y=-0.2, x=0.5, xanchor="center", yanchor="top"), margin=dict(b=120))


def _short(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(predicted, reference) yes/no labels, 1 = yes."""
    short = run["results"]["short_answer"]
    return np.array(short["predictions"]), np.array(short["references"])


def plot_accuracy(runs: dict[str, dict]) -> go.Figure:
    """Accuracy with its 95% bootstrap interval per method, and the McNemar p-value
    against the baseline (the first method)."""
    correct = {m: np.equal(*_short(run)) for m, run in runs.items()}
    reference = next(iter(runs))
    fig = go.Figure()
    for m, run in runs.items():
        accuracy = run["summary"]["accuracy"]
        low, high = run["confidence_intervals"]["accuracy"]
        fig.add_scatter(
            x=[accuracy], y=[METHODS[m]], mode="markers", showlegend=False,
            marker=dict(color=METHOD_COLOURS[m], size=13, line=dict(color="white", width=2)),
            error_x=dict(type="data", array=[high - accuracy], arrayminus=[accuracy - low],
                         color=METHOD_COLOURS[m], thickness=2.5, width=0),
            hovertemplate=f"{METHODS[m]}: %{{x:.1f}}% [{low:.1f}, {high:.1f}]<extra></extra>",
        )
        p = "reference" if m == reference else f"p = {mcnemar_p(correct[m], correct[reference]):.2g}"
        fig.add_annotation(x=1, xref="paper", xanchor="left", y=METHODS[m], text=p, showarrow=False,
                           font=dict(size=12, color=LIFE_HISTORIES[6]))
    fig.add_vline(x=runs[reference]["summary"]["accuracy"], line=dict(dash="dot", color=LIFE_HISTORIES[4], width=1.5))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Accuracy (%)", ticksuffix="")
    fig.update_layout(title=dict(text="Yes/no accuracy", subtitle_text=(
        "95% bootstrap intervals over the test questions; p: paired McNemar test against the baseline (dotted)")),
        height=360, margin=dict(r=110))
    return fig


def plot_nli(runs: dict[str, dict]) -> go.Figure:
    """Share of explanations that entail, are neutral to, or contradict the reference's."""
    fig = go.Figure()
    for label, colour in NLI_COLOURS.items():
        values = [run["summary"][label] for run in runs.values()]
        fig.add_bar(x=values, y=[METHODS[m] for m in runs], name=label.capitalize(), orientation="h",
                    marker=dict(color=colour), text=[f"{v:.0f}%" for v in values], textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color=LIFE_HISTORIES[0] if label == "neutral" else "white"),
                    hovertemplate=f"%{{y}}: %{{x:.1f}}% {label}<extra></extra>")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Explanations (%)", range=[0, 100])
    fig.update_layout(barmode="stack", bargap=0.35, height=380, legend=dict(traceorder="normal"), title=dict(
        text="Does the explanation agree with the reference?", subtitle_text=(
            "NLI (RoBERTa-large-MNLI): reference explanation as premise, model explanation as hypothesis")))
    return fig


def plot_training_history(trainings: dict[str, dict]) -> go.Figure:
    """Training and validation loss per adapter (one panel each), the kept checkpoint marked.

    ``trainings`` maps a method key to its loaded ``training.json``. The y-axis is zoomed
    on the validation loss, so the steep first steps of training run off the top.
    """
    fig = make_subplots(rows=1, cols=len(trainings), horizontal_spacing=0.08,
                        subplot_titles=[f"<b>{METHODS[m]}</b>" for m in trainings])
    for col, training in enumerate(trainings.values(), start=1):
        logs = training["log_history"]
        validation = [log["eval_loss"] for log in logs if "eval_loss" in log]
        floor = min(log["loss"] for log in logs if "loss" in log)
        fig.update_yaxes(range=[min(floor, *validation) - 0.02, max(validation) + 0.08], row=1, col=col)
        for key, label, colour in (("loss", "Training", METHOD_COLOURS["prompt_engineering"]),
                                   ("eval_loss", "Validation", METHOD_COLOURS["baseline"])):
            points = [(log["step"], log[key]) for log in logs if key in log]
            fig.add_scatter(x=[s for s, _ in points], y=[v for _, v in points], name=label, legendgroup=label,
                            showlegend=col == 1, mode="lines+markers" if key == "eval_loss" else "lines",
                            line=dict(color=colour, width=2), marker=dict(size=6), row=1, col=col)
        fig.add_vline(x=training["best_step"], line=dict(dash="dot", color=LIFE_HISTORIES[4], width=1.5),
                      row=1, col=col)
    fig.update_xaxes(title="Training step")
    fig.update_yaxes(title="Loss (answer tokens)", col=1)
    kept = ", ".join(f"{METHODS[m]} step {t['best_step']:,} ({t['best_eval_loss']:.4f})" for m, t in trainings.items())
    fig.update_layout(title=dict(text="Training history", subtitle_text=(
        f"Dotted: the checkpoint kept (lowest validation loss): {kept}. "
        "Zoomed on validation loss")), height=440, **PANEL_LAYOUT)
    return fig


def plot_calibration(runs: dict[str, dict], probabilities: dict[str, list[float]]) -> go.Figure:
    """Reliability diagram (10 equal-width bins) and ROC curve of P(yes) per method.

    ``probabilities`` maps a method key to its per-question P(yes), in test order.
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("<b>Calibration</b>", "<b>ROC curve</b>"))
    diagonal = dict(dash="dash", color=LIFE_HISTORIES[3], width=1.5)
    for col in (1, 2):
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=diagonal, hoverinfo="skip", showlegend=False,
                        row=1, col=col)
    brier = []
    for m, run in runs.items():
        p, y = np.array(probabilities[m]), _short(run)[1]
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
        fpr, tpr, _ = roc_curve(y, p)
        name = f"{METHODS[m]} (AUC {roc_auc_score(y, p):.3f})"
        style = dict(legendgroup=m, line=dict(color=METHOD_COLOURS[m], width=2))
        fig.add_scatter(x=predicted, y=observed, name=name, mode="lines+markers", marker=dict(size=6),
                        hovertemplate=f"{METHODS[m]}<br>predicted %{{x:.2f}}, observed %{{y:.2f}}<extra></extra>",
                        **style, row=1, col=1)
        fig.add_scatter(x=fpr, y=tpr, name=name, mode="lines", showlegend=False,
                        hovertemplate=f"{METHODS[m]}<br>FPR %{{x:.2f}}, TPR %{{y:.2f}}<extra></extra>",
                        **style, row=1, col=2)
        brier.append(f"{METHODS[m]} {brier_score_loss(y, p):.3f}")
    fig.update_xaxes(title="Predicted P(yes)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Observed share of yes", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title="False positive rate", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="True positive rate", range=[0, 1], row=1, col=2)
    fig.update_layout(title=dict(text="How trustworthy is P(yes)?", subtitle_text=(
        "Dashed: perfect calibration (left) and chance (right). Brier score (lower is better): " + ", ".join(brier))),
        height=500, **PANEL_LAYOUT)
    return fig


def plot_confusion(runs: dict[str, dict]) -> go.Figure:
    """2×2 confusion matrix per method: counts, with the share of each reference row."""
    fig = make_subplots(rows=1, cols=len(runs), horizontal_spacing=0.04,
                        subplot_titles=[f"<b>{METHODS[m]}</b>" for m in runs])
    matrices = {}
    for m, run in runs.items():
        predicted, reference = _short(run)
        matrices[m] = np.array([[np.sum((reference == r) & (predicted == p)) for p in (0, 1)] for r in (0, 1)])
    top = max(matrix.max() for matrix in matrices.values())
    for col, (m, matrix) in enumerate(matrices.items(), start=1):
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        fig.add_heatmap(z=matrix, x=["no", "yes"], y=["no", "yes"], coloraxis="coloraxis", xgap=3, ygap=3,
                        hovertemplate="reference %{y}, predicted %{x}: %{z}<extra></extra>", row=1, col=col)
        for r in (0, 1):
            for p in (0, 1):
                fig.add_annotation(x=p, y=r, text=f"<b>{matrix[r, p]}</b><br>{shares[r, p]:.0%}", showarrow=False,
                                   font=dict(color="white" if matrix[r, p] > top / 2 else LIFE_HISTORIES[0]),
                                   row=1, col=col)
    fig.update_xaxes(title="Predicted", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, showticklabels=False)
    fig.update_yaxes(title="Reference", showticklabels=True, col=1)
    fig.update_layout(title=dict(text="Where the yes/no answers go wrong", subtitle_text=(
        "Counts over the test questions; percentages are shares of each reference answer")),
        coloraxis=dict(colorscale=SEQUENTIAL, showscale=False), height=380, margin=dict(b=60))
    return fig


def plot_efficiency(runs: dict[str, dict], trainable: dict[str, int]) -> go.Figure:
    """Accuracy against trainable parameters (log scale) for the trained methods; methods
    that train nothing are drawn as horizontal lines with their 95% interval shaded.

    ``trainable`` maps each trained method key to its trainable-parameter count.
    """
    fig = go.Figure()
    for m, run in runs.items():
        accuracy = run["summary"]["accuracy"]
        low, high = run["confidence_intervals"]["accuracy"]
        colour = METHOD_COLOURS[m]
        if m in trainable:
            fig.add_scatter(x=[trainable[m]], y=[accuracy], mode="markers+text", text=[METHODS[m]],
                            textposition="middle right", showlegend=False, textfont=dict(color=colour),
                            marker=dict(color=colour, size=13, line=dict(color="white", width=2)),
                            error_y=dict(type="data", array=[high - accuracy], arrayminus=[accuracy - low],
                                         color=colour, thickness=2.5, width=0),
                            hovertemplate=(f"{METHODS[m]}: %{{y:.1f}}% with {trainable[m]:,} trainable "
                                           "parameters<extra></extra>"))
        else:
            fig.add_hrect(y0=low, y1=high, fillcolor=rgba(colour, 0.12), line_width=0)
            fig.add_hline(y=accuracy, line=dict(color=colour, width=2, dash="dash"),
                          annotation=dict(text=f"{METHODS[m]} (no training)", font=dict(color=colour),
                                          xanchor="left", x=0, yanchor="bottom"))
    fig.update_xaxes(type="log", title="Trainable parameters (log scale)", range=[5.8, 9.2])
    fig.update_yaxes(title="Accuracy (%)")
    fig.update_layout(title=dict(text="Accuracy against trainable parameters", subtitle_text=(
        "95% bootstrap intervals; shaded bands: intervals of the methods without training")), height=440)
    return fig
