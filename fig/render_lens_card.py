"""The lens card: one torn question, the same 16 layers, three readouts —
raw logit lens, tuned lens, J-lens — each with a one-line definition in the
header. Cells show the top-2 tokens (big + small). The figure that answers
"what are these lenses" without sending anyone to a paper.

Data: fig/figdata_gallery.json (per-case top-2 grids for all three lenses).

uv run --with matplotlib --no-project python3 fig/render_lens_card.py [cz]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{HERE}/figdata_gallery.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_CALC, C_WEB = "#c98500", "#3987e5"
LAYERS = range(20, 36)
CASE = "torn_leap_ms"

CZ = "cz" in sys.argv[1:]
T = {
    "title": "Three ways to read the same moment",
    "sub": "one question, the token where the agent writes the tool name, layers 21–36 · top-2 tokens per layer",
    "q": "“How many milliseconds are in a leap year?”  →  calls web_search",
    "cols": [
        ("lens", "raw logit lens", "each layer decoded directly through the\noutput head — simple, but uncalibrated\nmid-stack", "no training"),
        ("tlens", "tuned lens", "the same readout with a trained per-layer\ncorrection — what the layer would say now,\ncalibrated (Belrose et al. 2023)", "fit: wikitext-103"),
        ("jlens", "J-lens", "activations mapped by a fitted Jacobian —\nwhat the layer pushes the model to say\nLATER (Anthropic 2026)", "fit: averaged Jacobian"),
    ],
    "verdict": "all three readouts hold “calculator” deep into the stack — the J-lens at p≈1.0 one layer before the end —\nand the model calls web_search. The JSON that comes out would never tell you.",
    "foot": "Qwen3-4B · greedy, tool_choice: required · exact per-case grids: fig/extract.py · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Tři způsoby, jak přečíst tentýž okamžik",
        "sub": "jedna otázka, token kde agent píše jméno toolu, vrstvy 21–36 · top-2 tokeny na vrstvu",
        "q": "„How many milliseconds are in a leap year?“  →  volá web_search",
        "cols": [
            ("lens", "raw logit lens", "každá vrstva dekódovaná přímo přes\nvýstupní hlavu — jednoduché, ale uprostřed\nsítě nekalibrované", "bez tréninku"),
            ("tlens", "tuned lens", "týž readout s natrénovanou korekcí po\nvrstvách — co by vrstva řekla teď,\nkalibrovaně (Belrose et al. 2023)", "fit: wikitext-103"),
            ("jlens", "J-lens", "aktivace zobrazené fitnutým Jacobiánem —\nk čemu vrstva tlačí model POZDĚJI\n(Anthropic 2026)", "fit: průměrovaný Jacobián"),
        ],
        "verdict": "všechny tři readouty drží „calculator“ hluboko do sítě — J-lens p≈1,0 vrstvu před koncem —\na model zavolá web_search. Ze stejného JSONu to nepoznáš.",
        "foot": "Qwen3-4B · greedy, tool_choice: required · exaktní per-case gridy: fig/extract.py · brainscope — github.com/moudrkat/brainscope",
    }


def classify(tok):
    t = tok.strip().lstrip('"').strip().lower()
    if len(t) < 2:
        return None
    if "calculator".startswith(t) or t in ("calculate", "calculator"):
        return "calc"
    if "web_search".startswith(t) or t == "web":
        return "web"
    return None


d = DATA[CASE]
COL_W, CELL_H, GAP = 3.1, 0.5, 0.55
n_rows = len(LAYERS)
x0, y0 = 1.15, 1.55
fig_w = x0 + 3 * COL_W + 2 * GAP + 0.4
fig_h = y0 + n_rows * CELL_H + 3.4
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
fig.patch.set_facecolor(PAGE)
ax.set_facecolor(PAGE)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

top = y0 + n_rows * CELL_H
ax.text(fig_w / 2, fig_h - 0.40, T["title"], ha="center", color=INK,
        fontsize=17, fontweight="bold", family="sans-serif")
ax.text(fig_w / 2, fig_h - 0.78, T["sub"], ha="center", color=INK2,
        fontsize=10, family="sans-serif")
ax.text(fig_w / 2, fig_h - 1.12, T["q"], ha="center", color=C_WEB,
        fontsize=11, family="monospace", fontweight="bold")

for r, layer in enumerate(LAYERS):
    y = y0 + r * CELL_H
    if (layer + 1) % 2 == 0:
        ax.text(x0 - 0.12, y + CELL_H / 2, f"L{layer + 1}", ha="right",
                va="center", color=MUTED, fontsize=8.5, family="monospace")

for ci, (key, name_, desc, prov) in enumerate(T["cols"]):
    cx = x0 + ci * (COL_W + GAP)
    ax.text(cx + COL_W / 2, top + 1.62, name_, ha="center", color=INK,
            fontsize=12.5, fontweight="bold", family="sans-serif")
    ax.text(cx + COL_W / 2, top + 1.02, desc, ha="center", va="center",
            color=INK2, fontsize=8, family="sans-serif", linespacing=1.35)
    ax.text(cx + COL_W / 2, top + 0.42, prov, ha="center", color=MUTED,
            fontsize=7.5, family="monospace")
    for r, layer in enumerate(LAYERS):
        y = y0 + r * CELL_H
        entries = d[key][layer][:2]
        e = entries[0]
        tool = classify(e["t"])
        p = e["p"]
        if tool == "calc":
            fc, a, tc = C_CALC, 0.18 + 0.72 * p, INK
        elif tool == "web":
            fc, a, tc = C_WEB, 0.18 + 0.72 * p, INK
        else:
            fc, a, tc = SURFACE, 1.0, MUTED
        ax.add_patch(FancyBboxPatch(
            (cx, y + 0.035), COL_W, CELL_H - 0.07,
            boxstyle="round,pad=0,rounding_size=0.06",
            fc=fc, ec=GRID, lw=0.6, alpha=a))
        tok = e["t"].strip()
        if not all(ord(c) < 0x2500 for c in tok):
            tok = "···"
        ax.text(cx + 0.12, y + CELL_H / 2, tok[:14], ha="left", va="center",
                color=tc, fontsize=10, family="monospace",
                fontweight="bold" if tool else "normal")
        if tool and p >= 0.2:
            ax.text(cx + COL_W - 1.05, y + CELL_H / 2, f"{p:.2f}", ha="right",
                    va="center", color=tc, fontsize=8, family="monospace",
                    alpha=0.9)
        if len(entries) > 1:
            t2 = entries[1]["t"].strip()
            if not all(ord(c) < 0x2500 for c in t2):
                t2 = "···"
            c2 = classify(entries[1]["t"])
            ax.text(cx + COL_W - 0.10, y + CELL_H / 2, t2[:9], ha="right",
                    va="center",
                    color=(C_CALC if c2 == "calc" else C_WEB if c2 == "web"
                           else MUTED),
                    fontsize=7, family="monospace", alpha=0.85)

ax.text(fig_w / 2, y0 - 0.42, T["verdict"], ha="center", va="top",
        color=INK2, fontsize=9.8, family="sans-serif", linespacing=1.5)
ax.text(0.25, 0.25, T["foot"], color=MUTED, fontsize=8.2, family="sans-serif")

name = "fig_lens_card_" + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.25)
print("saved", name)
