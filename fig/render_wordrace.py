"""The word race: every token is written straight into the chart at the
layer where its mean probability peaks, sized by that peak; a thin line
traces its whole curve. No legend, no bars — just words rising and falling
with depth until the schema name outgrows everything.

Data: stored top-8 tuned-lens readouts (mean over cases = a lower bound).

uv run --with matplotlib --no-project python3 fig/render_wordrace.py [cz]
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{HERE}/vocab_census.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_CALC, C_WEB, C_GEN = "#c98500", "#3987e5", "#2fa889"
GRAY = "#6a7590"

CZ = "cz" in sys.argv[1:]
T = {
    "title": "The word race — what the readout says on the way to the tool name",
    "sub": "each word sits at the layer where its mean probability peaks, sized by that peak · "
           "tuned lens at the tool-name token, 815 questions (top-8 readouts, a lower bound)",
    "web": "cases that called web_search (n=%d)",
    "calc": "cases that called calculator (n=%d)",
    "layer": "layer (depth →)",
    "meanp": "mean p",
    "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023) · battery: vocab.py · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Slovní závod — co říká readout cestou ke jménu toolu",
        "sub": "každé slovo sedí ve vrstvě, kde vrcholí jeho průměrné p, velikost = ten vrchol · "
               "tuned lens na tokenu jména toolu, 815 otázek (top-8 readouty, dolní odhad)",
        "web": "cases volající web_search (n=%d)",
        "calc": "cases volající calculator (n=%d)",
        "layer": "vrstva (hloubka →)",
        "meanp": "průměrné p",
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


def curves(cases, min_peak=0.02, max_words=18):
    acc = defaultdict(lambda: [0.0] * 36)
    n = len(cases)
    for v in cases.values():
        for l in range(36):
            for e in v["top8"]["tuned"][l]:
                t = e["t"].strip().lstrip('"').strip().lower()
                if len(t) >= 2 and all(ord(c) < 0x2500 for c in t):
                    acc[t][l] += e["p"] / n
    peaked = {t: c for t, c in acc.items() if max(c) >= min_peak}
    keep = sorted(peaked, key=lambda t: -max(peaked[t]))[:max_words]
    return {t: peaked[t] for t in keep}


web = {k: v for k, v in DATA.items() if v["picked"] == "web_search"}
calc = {k: v for k, v in DATA.items() if v["picked"] == "calculator"}

fig, axes = plt.subplots(2, 1, figsize=(13.4, 10.6), dpi=160,
                         gridspec_kw={"hspace": 0.32})
fig.patch.set_facecolor(PAGE)
fig.suptitle(T["title"], color=INK, fontsize=17, fontweight="bold",
             family="sans-serif", y=0.975)
fig.text(0.5, 0.925, T["sub"], ha="center", color=INK2, fontsize=9.8,
         family="sans-serif")

for ax, cases, ptitle in [(axes[0], web, T["web"]), (axes[1], calc, T["calc"])]:
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(1, 36.8)
    ax.set_ylim(0, 1.02)
    ax.set_title(ptitle % len(cases), color=INK2, fontsize=11,
                 family="sans-serif", loc="left", pad=8)
    ax.set_ylabel(T["meanp"], color=MUTED, fontsize=9.5)
    cs = curves(cases)
    placed = []      # (x, y) of placed labels, for simple collision nudge
    for t, c in sorted(cs.items(), key=lambda kv: max(kv[1])):
        color = tok_color(t)
        xs = range(1, 37)
        ax.plot(xs, c, color=color, lw=1.1, alpha=0.45)
        pl = max(range(36), key=lambda l: c[l])
        px, py = pl + 1, max(c)
        size = 8 + 26 * min(1.0, py / 0.9)
        # nudge straight up if a previous label is too close
        for qx, qy in placed:
            if abs(px - qx) < 2.6 and abs(py - qy) < 0.055:
                py += 0.055
        placed.append((px, py))
        ax.text(px, py + 0.015, t, ha="center", va="bottom", color=color,
                fontsize=size, family="monospace",
                fontweight="bold" if py > 0.10 else "normal")
axes[1].set_xlabel(T["layer"], color=MUTED, fontsize=9.5)
fig.text(0.01, 0.005, T["foot"], color=MUTED, fontsize=8.2, family="sans-serif")

name = "fig_wordrace_" + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.3)
print("saved", name)
