"""The eternal runner-up: `google` under two readouts. In the tuned lens
census it appears 207 times in the top-8 and never once wins; in the J-lens
it is the top-1 through layers ~21–31 in most gallery cases. Same state,
two questions, two different favorite words.

uv run --with matplotlib --no-project python3 fig/render_eternal_second.py [cz]
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
CENSUS = json.load(open(f"{HERE}/vocab_census.json"))
GALLERY = json.load(open(f"{HERE}/figdata_gallery.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_WEB, C_GOOGLE, C_Q, C_S = "#3987e5", "#e0a33d", "#2fa889", "#57c3a5"

CZ = "cz" in sys.argv[1:]
T = {
    "title": "The eternal runner-up",
    "sub": "the word “google” under two readouts of the same hidden states",
    "left": "tuned lens — “would say now, corrected”\nmean p over 450 web-bound questions",
    "left_note": "google: 207× in the top-8, 0× first —\nit always loses to query or search",
    "right": "J-lens — “pushed toward later”\ntop-1 token by layer, gallery case torn_gdp",
    "right_note": "under the J-lens, google IS the winner\nthrough the middle of the stack\n(top-1 at L21–31 in 8 of 12 gallery cases)",
    "layer": "layer (depth →)",
    "meanp": "mean p",
    "foot": "Qwen3-4B · census: vocab.py (tuned lens, top-8 lower bound) · gallery: agent.py + fig/extract.py · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Věčná dvojka",
        "sub": "slovo „google“ pod dvěma readouty týchž hidden states",
        "left": "tuned lens — „co by řekl teď, opraveno“\nprůměrné p přes 450 web-bound otázek",
        "left_note": "google: 207× v top-8, 0× první —\nvždy prohraje s query nebo search",
        "right": "J-lens — „k čemu tlačí později“\ntop-1 token po vrstvách, gallery case torn_gdp",
        "right_note": "pod J-lens google VYHRÁVÁ\nskrz střed sítě\n(top-1 na L21–31 v 8 z 12 gallery cases)",
        "layer": "vrstva (hloubka →)",
        "meanp": "průměrné p",
        "foot": "Qwen3-4B · census: vocab.py (tuned lens, top-8 dolní odhad) · gallery: agent.py + fig/extract.py · brainscope — github.com/moudrkat/brainscope",
    }

web = {k: v for k, v in CENSUS.items() if v["picked"] == "web_search"}
acc = defaultdict(lambda: [0.0] * 36)
for v in web.values():
    for l in range(36):
        for e in v["top8"]["tuned"][l]:
            t = e["t"].strip().lstrip('"').strip().lower()
            acc[t][l] += e["p"] / len(web)

fig = plt.figure(figsize=(13.0, 5.6), dpi=160)
fig.patch.set_facecolor(PAGE)
fig.suptitle(T["title"], color=INK, fontsize=17, fontweight="bold",
             family="sans-serif", y=0.97)
fig.text(0.5, 0.885, T["sub"], ha="center", color=INK2, fontsize=10.5,
         family="sans-serif")

ax = fig.add_axes((0.07, 0.14, 0.52, 0.60))
ax.set_facecolor(PAGE)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.grid(color=GRID, lw=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.set_xlim(18, 36)
ax.set_ylim(0, 0.42)
ax.set_title(T["left"], color=INK2, fontsize=9.5, family="sans-serif",
             loc="left", pad=8, linespacing=1.4)
ax.set_xlabel(T["layer"], color=MUTED, fontsize=9)
ax.set_ylabel(T["meanp"], color=MUTED, fontsize=9)
for tok, color, lw in [("query", C_Q, 2), ("search", C_S, 2),
                       ("web", C_WEB, 1.2), ("google", C_GOOGLE, 2.6)]:
    ys = acc[tok]
    ax.plot(range(1, 37), ys, color=color, lw=lw,
            alpha=1.0 if tok == "google" else 0.75)
    pl = max(range(36), key=lambda l: ys[l])
    if 18 <= pl + 1 <= 36:
        ax.annotate(tok, xy=(pl + 1, ys[pl]),
                    xytext=(pl + 1, min(0.40, ys[pl] + 0.015)),
                    ha="center", color=color, fontsize=10,
                    family="monospace", fontweight="bold")
ax.text(0.03, 0.95, T["left_note"], transform=ax.transAxes, va="top",
        color=INK2, fontsize=8.8, family="sans-serif", linespacing=1.4)

# right: the J-lens strip for torn_gdp — google as literal top-1
axr = fig.add_axes((0.66, 0.14, 0.30, 0.60))
axr.set_facecolor(PAGE)
axr.axis("off")
axr.set_title(T["right"], color=INK2, fontsize=9.5, family="sans-serif",
              loc="left", pad=8, linespacing=1.4)
g = GALLERY["torn_gdp"]["jlens"]
rows = list(range(20, 36))
for i, l in enumerate(rows):
    y = 1 - (i + 1) / len(rows)
    e = g[l][0]
    tok = e["t"].strip().lstrip('"').strip().lower() or "·"
    is_g = tok == "google"
    is_web = tok in ("web", "web_search")
    fc = C_GOOGLE if is_g else C_WEB if is_web else SURFACE
    axr.add_patch(FancyBboxPatch(
        (0.14, y), 0.55, 0.85 / len(rows),
        boxstyle="round,pad=0,rounding_size=0.01",
        fc=fc, ec=GRID, lw=0.5,
        alpha=0.9 if (is_g or is_web) else 1.0,
        transform=axr.transAxes))
    axr.text(0.16, y + 0.42 / len(rows), tok[:12], transform=axr.transAxes,
             va="center", color=INK if (is_g or is_web) else MUTED,
             fontsize=8.5, family="monospace",
             fontweight="bold" if (is_g or is_web) else "normal")
    axr.text(0.11, y + 0.42 / len(rows), f"L{l + 1}", transform=axr.transAxes,
             va="center", ha="right", color=MUTED, fontsize=7,
             family="monospace")
axr.text(0.72, 0.55, T["right_note"], transform=axr.transAxes, va="center",
         color=INK2, fontsize=8.8, family="sans-serif", linespacing=1.45)

fig.text(0.01, 0.01, T["foot"], color=MUTED, fontsize=8, family="sans-serif")
name = "fig_eternal_second_" + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.3)
print("saved", name)
