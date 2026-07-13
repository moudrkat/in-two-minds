"""Podium by depth: at each layer, the three tokens with the highest mean
probability (tuned lens, from stored top-8 readouts — a lower bound) as
three lanes: 1st, 2nd, 3rd. The eternal runner-up is visible at a glance.

uv run --with matplotlib --no-project python3 fig/render_podium.py [cz]
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{HERE}/vocab_census.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_CALC, C_WEB, C_GEN = "#c98500", "#3987e5", "#2fa889"
GRAY = "#465071"

CZ = "cz" in sys.argv[1:]
T = {
    "title": "The podium — the three most likely tokens at every depth",
    "sub": "mean probability at the tool-name token under the tuned lens, 815 questions · "
           "cell shade = mean p · from stored top-8 readouts (a lower bound)",
    "web": "cases that called web_search (n=%d)",
    "calc": "cases that called calculator (n=%d)",
    "lanes": ["1st", "2nd", "3rd"],
    "layer": "layer (depth →)",
    "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023) · battery: vocab.py · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Pódium — tři nejpravděpodobnější tokeny v každé hloubce",
        "sub": "průměrné p na tokenu jména toolu pod tuned lens, 815 otázek · "
               "sytost buňky = průměrné p · z uložených top-8 readoutů (dolní odhad)",
        "web": "cases volající web_search (n=%d)",
        "calc": "cases volající calculator (n=%d)",
        "lanes": ["1.", "2.", "3."],
        "layer": "vrstva (hloubka →)",
        "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023) · battery: vocab.py · brainscope — github.com/moudrkat/brainscope",
    }

GENERIC = {"search", "lookup", "look", "query", "google", "api", "database",
           "find", "fetch", "browse", "retrieve", "internet", "calculate",
           "calc", "calculation", "compute", "computing", "math", "evaluate",
           "formula", "conversion", "function"}


def tok_color(t):
    if t == "calculator":
        return C_CALC
    if t in ("web", "web_search"):
        return C_WEB
    if t in GENERIC:
        return C_GEN
    return GRAY


def podium(cases):
    """per layer: [(token, mean_p) x3] by mean p over cases (top-8 lower bound)"""
    out = []
    n = len(cases)
    for l in range(36):
        acc = defaultdict(float)
        for v in cases.values():
            for e in v["top8"]["tuned"][l]:
                t = e["t"].strip().lstrip('"').strip().lower()
                if len(t) >= 2:
                    acc[t] += e["p"]
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:3]
        out.append([(t, p / n) for t, p in top])
    return out


web = {k: v for k, v in DATA.items() if v["picked"] == "web_search"}
calc = {k: v for k, v in DATA.items() if v["picked"] == "calculator"}

CELL_W, CELL_H = 0.62, 0.58
fig_w = 1.6 + 36 * CELL_W + 0.4
fig_h = 2.0 + 2 * (3 * CELL_H + 1.35) + 0.8
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
fig.patch.set_facecolor(PAGE)
ax.set_facecolor(PAGE)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

ax.text(fig_w / 2, fig_h - 0.42, T["title"], ha="center", color=INK,
        fontsize=17, fontweight="bold", family="sans-serif")
ax.text(fig_w / 2, fig_h - 0.82, T["sub"],
        ha="center", color=INK2, fontsize=10, family="sans-serif")

x0 = 1.5
for pi, (cases, ptitle) in enumerate([(web, T["web"]), (calc, T["calc"])]):
    py = fig_h - 1.9 - pi * (3 * CELL_H + 1.35)
    ax.text(x0, py + 0.32, ptitle % len(cases), ha="left", color=INK2,
            fontsize=11, family="sans-serif")
    pod = podium(cases)
    for lane in range(3):
        y = py - (lane + 1) * CELL_H
        ax.text(x0 - 0.15, y + CELL_H / 2, T["lanes"][lane], ha="right",
                va="center", color=MUTED, fontsize=10, family="monospace")
        run_start, run_tok = 0, pod[0][lane][0] if len(pod[0]) > lane else ""
        for l in range(37):
            tok = pod[l][lane][0] if l < 36 and len(pod[l]) > lane else None
            if l == 36 or tok != run_tok:
                # draw the finished run as one merged cell
                if run_tok:
                    if not all(ord(c) < 0x2500 for c in run_tok):
                        run_tok = "···"
                    ps = [pod[j][lane][1] for j in range(run_start, l)
                          if len(pod[j]) > lane]
                    p = max(ps) if ps else 0
                    cx = x0 + run_start * CELL_W
                    w = (l - run_start) * CELL_W
                    a = 0.15 + 0.85 * min(1.0, p / 0.6)
                    ax.add_patch(FancyBboxPatch(
                        (cx, y + 0.04), w - 0.06, CELL_H - 0.08,
                        boxstyle="round,pad=0,rounding_size=0.06",
                        fc=tok_color(run_tok), ec=GRID, lw=0.5, alpha=a))
                    label = run_tok if len(run_tok) * 0.11 < w else \
                        run_tok[:max(1, int(w / 0.11)) - 1] + "…" if w > 0.35 else ""
                    if label:
                        ax.text(cx + w / 2 - 0.03, y + CELL_H / 2, label,
                                ha="center", va="center", color=INK,
                                fontsize=9 if len(label) * 0.105 < w else 7.5,
                                family="monospace",
                                fontweight="bold" if p > 0.15 else "normal")
                if l < 36:
                    run_start, run_tok = l, tok
    for l in (0, 4, 9, 14, 19, 24, 29, 35):
        ax.text(x0 + l * CELL_W + CELL_W / 2, py - 3 * CELL_H - 0.22,
                f"L{l + 1}", ha="center", color=MUTED, fontsize=8,
                family="monospace")
    ax.text(x0 + 18 * CELL_W, py - 3 * CELL_H - 0.52, T["layer"],
            ha="center", color=MUTED, fontsize=9.5, family="sans-serif")

ax.text(0.25, 0.25, T["foot"], color=MUTED, fontsize=8.5, family="sans-serif")
name = "fig_podium_" + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.25)
print("saved", name)
