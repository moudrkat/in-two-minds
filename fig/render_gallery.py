"""Gallery figure: every case × three lens columns (raw | tuned | J-lens),
verdict per case computed from the data. Run fig/extract.py and
fig/tlens_extract.py first, merge into figdata_gallery.json, then:
uv run --with matplotlib --no-project python3 fig/render_gallery.py [cz]

Single-lens variants (verdicts then use only the shown lens):
uv run --with matplotlib --no-project python3 fig/render_gallery.py [cz] logit|tuned|jlens
"""
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import os
SCRATCH = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{SCRATCH}/figdata_gallery.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_CALC, C_WEB = "#c98500", "#3987e5"
LAYERS = range(20, 36)          # 0-based -> shown as L21..L36

CZ = "cz" in sys.argv[1:]
T = {
    "title": "Twelve questions, three lenses each — where the agent is sure, and where it wavers",
    "subtitle": "per-layer readout at the token where the agent writes the tool name · "
                "columns: raw logit lens | tuned lens (Belrose et al. 2023) | J-lens (Anthropic 2026)",
    "leg_calc": "reads as calculator", "leg_web": "reads as web_search",
    "leg_none": "neither  ·  shade = probability",
    "calls": "calls",
    "sure": "SURE", "hesitates": "HESITATES", "changes": "CHANGES ITS MIND",
    "cols": ["raw", "tuned", "J"],
    "foot": "Qwen3-4B · greedy, tool_choice: required — every call returns identical, schema-perfect JSON · "
            "rows L21–36 (layers 1–20: no tool signal) · verdicts computed from the readouts · "
            "this illustrates; measuring means intervening · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Dvanáct otázek, tři lens na každou — kde si je agent jistý a kde kolísá",
        "subtitle": "čtení po vrstvách na tokenu, kde agent píše jméno toolu · "
                    "sloupce: raw logit lens | tuned lens (Belrose et al. 2023) | J-lens (Anthropic 2026)",
        "leg_calc": "čte se jako calculator", "leg_web": "čte se jako web_search",
        "leg_none": "nic z toho  ·  sytost = pravděpodobnost",
        "calls": "volá",
        "sure": "JISTÝ", "hesitates": "VÁHÁ", "changes": "ROZMYSLÍ SI TO",
        "cols": ["raw", "tuned", "J"],
        "foot": "Qwen3-4B · greedy, tool_choice: required — každý call vrací identický, validní JSON · "
                "řádky L21–36 (vrstvy 1–20: po toolech ani stopa) · verdikty spočtené z readoutů · "
                "tohle ilustruje; měřit znamená zasáhnout · brainscope — github.com/moudrkat/brainscope",
    }

ONLY = next((a for a in sys.argv[1:] if a in ("logit", "tuned", "jlens")), None)
KEYS = {"logit": ["lens"], "tuned": ["tlens"], "jlens": ["jlens"]}[ONLY] if ONLY \
    else ["lens", "tlens", "jlens"]
if ONLY:
    SOLO = {
        "logit": ("raw logit lens", "what each layer would say if generation stopped there",
                  "raw logit lens", "co by každá vrstva řekla, kdyby generování skončilo tam"),
        "tuned": ("tuned lens (Belrose et al. 2023)", "the logit lens with trained per-layer corrections",
                  "tuned lens (Belrose et al. 2023)", "logit lens s natrénovanou korekcí po vrstvách"),
        "jlens": ("J-lens (Anthropic 2026)", "what each layer pushes the model to say later",
                  "J-lens (Anthropic 2026)", "k čemu každá vrstva tlačí model později"),
    }[ONLY]
    T["title"] = ("Twelve questions through the " if not CZ else "Dvanáct otázek přes ") \
        + (SOLO[0] if not CZ else SOLO[2])
    T["subtitle"] = ("per-layer readout at the token where the agent writes the tool name · "
                     if not CZ else
                     "čtení po vrstvách na tokenu, kde agent píše jméno toolu · ") \
        + (SOLO[1] if not CZ else SOLO[3])


def classify(tok: str) -> str | None:
    t = tok.strip().lstrip('"').strip().lower()
    if len(t) < 2:
        return None
    if "calculator".startswith(t) or t in ("calculate", "calculator"):
        return "calc"
    if "web_search".startswith(t) or t == "web":
        return "web"
    return None


def verdict(d) -> tuple[str, str]:
    """(label, color) from the readouts: rival tool on top at L34–35 in any
    lens = changed its mind at the end; rival on top anywhere at L26+ =
    hesitated; otherwise sure."""
    final = classify(d["lens"][35][0]["t"])
    late = mid = False
    for key in KEYS:
        for l in range(25, 35):
            tool = classify(d[key][l][0]["t"])
            if tool and tool != final:
                if l >= 33:
                    late = True
                else:
                    mid = True
    if late:
        return T["changes"], "#e05d5d"
    if mid:
        return T["hesitates"], "#e0a33d"
    return T["sure"], "#4caf7d"


ORDER = [c for c in (
    "calc_clear", "calc_clear2", "search_clear", "search_clear2",
    "torn_boiling", "torn_leap_ms", "torn_eiffel", "torn_dog",
    "torn_sunlight", "torn_moon", "torn_marathon", "torn_gdp") if c in DATA]
NCOL = 4
NROW = (len(ORDER) + NCOL - 1) // NCOL

COL_W, CELL_H = (2.2, 0.30) if ONLY else (1.28, 0.30)
PANEL_W = len(KEYS) * COL_W + (len(KEYS) - 1) * 0.06
PANEL_GAP_X, PANEL_GAP_Y = (1.5, 1.05) if ONLY else (0.85, 1.05)
QWRAP = 30 if ONLY else 34
HEAD_H = 1.55                                # question + chip + col labels
GRID_H = len(LAYERS) * CELL_H
PANEL_H = HEAD_H + GRID_H + 0.55             # + verdict line

fig_w = 1.7 + NCOL * PANEL_W + (NCOL - 1) * PANEL_GAP_X + 0.6
fig_h = 2.1 + NROW * PANEL_H + (NROW - 1) * PANEL_GAP_Y + 0.9
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
fig.patch.set_facecolor(PAGE)
ax.set_facecolor(PAGE)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

ax.text(fig_w / 2, fig_h - 0.50, T["title"],
        ha="center", color=INK, fontsize=19, fontweight="bold", family="sans-serif")
ax.text(fig_w / 2, fig_h - 0.95, T["subtitle"],
        ha="center", color=INK2, fontsize=11.5, family="sans-serif")

leg = [(C_CALC, T["leg_calc"]), (C_WEB, T["leg_web"]), (SURFACE, T["leg_none"])]
lw_total = sum(0.40 + 0.088 * len(lab) + 0.5 for _, lab in leg)
lx = (fig_w - lw_total) / 2
ly = fig_h - 1.38
for c, lab in leg:
    ax.add_patch(FancyBboxPatch((lx, ly - 0.09), 0.28, 0.19,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc=c, ec=GRID, lw=0.8))
    ax.text(lx + 0.40, ly, lab, ha="left", va="center", color=INK2,
            fontsize=9.5, family="sans-serif")
    lx += 0.40 + 0.088 * len(lab) + 0.5

x00, y00 = 1.35, 0.95

for idx, case in enumerate(ORDER):
    d = DATA[case]
    row, col = divmod(idx, NCOL)
    px = x00 + col * (PANEL_W + PANEL_GAP_X)
    py = y00 + (NROW - 1 - row) * (PANEL_H + PANEL_GAP_Y)
    gy = py + 0.55                            # grid bottom
    gt = gy + GRID_H                          # grid top

    qlines = textwrap.wrap("“" + d["q"] + "”", QWRAP)[:3]
    for li, line in enumerate(qlines):
        ax.text(px + PANEL_W / 2, gt + HEAD_H - 0.10 - li * 0.26, line,
                ha="center", va="center", color=INK, fontsize=9.5,
                family="sans-serif", fontweight="bold")
    picked = classify(d["lens"][35][0]["t"])
    name = "calculator" if picked == "calc" else "web_search"
    chip_c = C_CALC if picked == "calc" else C_WEB
    ax.text(px + PANEL_W / 2, gt + 0.62, f"{T['calls']} {name}",
            ha="center", va="center", color=chip_c, fontsize=8.5,
            family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc=SURFACE, ec=chip_c, lw=1.1))

    for ci, key in enumerate(KEYS):
        cx = px + ci * (COL_W + 0.06)
        if not ONLY:
            ax.text(cx + COL_W / 2, gt + 0.18, T["cols"][ci], ha="center",
                    va="center", color=MUTED, fontsize=8, family="sans-serif")
        for r, layer in enumerate(LAYERS):
            y = gy + r * CELL_H
            e = d[key][layer][0]
            tool = classify(e["t"])
            p = e["p"]
            if tool == "calc":
                fc, a, tc = C_CALC, 0.18 + 0.72 * p, INK
            elif tool == "web":
                fc, a, tc = C_WEB, 0.18 + 0.72 * p, INK
            else:
                fc, a, tc = SURFACE, 1.0, MUTED
            ax.add_patch(FancyBboxPatch(
                (cx, y + 0.025), COL_W, CELL_H - 0.05,
                boxstyle="round,pad=0,rounding_size=0.05",
                fc=fc, ec=GRID, lw=0.5, alpha=a, mutation_aspect=1))
            tok = e["t"].strip()
            if not all(ord(c) < 0x2500 for c in tok):
                tok = "···"
            maxlen = 13 if ONLY else 9
            if len(tok) > maxlen:
                tok = tok[:maxlen - 1] + "…"
            if ONLY and tool and p >= 0.3:
                ax.text(cx + COL_W - 0.08, y + CELL_H / 2, f"{p:.2f}", ha="right",
                        va="center", color=tc, fontsize=6.5, family="monospace", alpha=0.85)
            ax.text(cx + 0.08, y + CELL_H / 2, tok, ha="left", va="center",
                    color=tc, fontsize=7, family="monospace",
                    fontweight="bold" if tool else "normal")
        if ci == 0:                            # layer ticks on panel's left
            for r, layer in enumerate(LAYERS):
                if (layer + 1) % 5 == 0 or layer in (LAYERS[0], 35):
                    ax.text(cx - 0.08, gy + r * CELL_H + CELL_H / 2,
                            f"L{layer + 1}", ha="right", va="center",
                            color=MUTED, fontsize=6.5, family="monospace")

    label, vc = verdict(d)
    ax.text(px + PANEL_W / 2, gy - 0.28, label, ha="center", va="center",
            color=vc, fontsize=9.5, family="sans-serif", fontweight="bold")

for fi, fline in enumerate(textwrap.wrap(T["foot"], int((fig_w - 1.2) / 0.075))):
    ax.text(x00 - 0.9, 0.52 - fi * 0.28, fline, color=MUTED, fontsize=9, family="sans-serif")

name = "fig_gallery_" + (ONLY + "_" if ONLY else "") + ("cz" if CZ else "en")
fig.savefig(f"{SCRATCH}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.25)
print("saved", name)
