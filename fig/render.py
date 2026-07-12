"""LinkedIn figure: three questions, six lens columns, one story."""

# deps: matplotlib only — run e.g.  uv run --with matplotlib --no-project python3 fig/render.py [cz]
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import os
SCRATCH = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{SCRATCH}/figdata.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_CALC, C_WEB = "#c98500", "#3987e5"
LAYERS = range(20, 36)          # 0-based -> shown as L21..L36

CZ = "cz" in sys.argv[1:]
LOGIT_ONLY = "logit" in sys.argv[1:]   # single-lens variant of the figure
TUNED = "tuned" in sys.argv[1:]        # three-lens variant: raw | tuned | J-lens

T = {
    "title": "Three questions. Three clean tool calls. One changed its mind at the last layer.",
    "subtitle": "An agent picks between calculator and web_search — per-layer readout at the moment it writes the tool name",
    "leg_calc": "reads as calculator", "leg_web": "reads as web_search",
    "leg_none": "neither  ·  shade = probability",
    "layers_note": "⋮ layers 1–20: no tool signal yet", "depth": "layer (depth →)",
    "calls": "calls",
    "lens_label": "logit lens\n“would say now”", "jlens_label": "J-lens\n“pushed toward later”",
    "tlens_label": "tuned lens\n“corrected readout”",
    "v1": "SURE — from layer 27 up, both\nreadouts have calculator on top\nand it never changes.",
    "v2": "HESITATES — mid-stack it reaches for\n“lookup”, a tool it doesn't even have,\nthen flip-flops calculator ↔ web.",
    "v3": "CHANGES ITS MIND — at layer 35 of 36\nthe J-lens still says “calculator is coming”\n(p ≈ 1.0); the last layer overrides it.",
    "callout1": "one layer before the output,\nstill certain the answer\nis “calculator”",
    "callout2": "the last layer flips it",
    "foot1": "logit lens: what the model would emit if it stopped at this layer   ·   "
             "J-lens (Jacobian lens, Anthropic 2026): what this layer is pushing the model to say later",
    "foot2": "Qwen3-4B · greedy, tool_choice: required — every call returns identical, schema-perfect JSON · "
             "this illustrates; measuring means intervening (steer & rerun) · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Tři otázky. Tři čisté tool cally. Jeden si to rozmyslel v poslední vrstvě.",
        "subtitle": "Agent volí mezi calculator a web_search — čtení po vrstvách ve chvíli, kdy píše jméno toolu",
        "leg_calc": "čte se jako calculator", "leg_web": "čte se jako web_search",
        "leg_none": "nic z toho  ·  sytost = pravděpodobnost",
        "layers_note": "⋮ vrstvy 1–20: po toolech ani stopa", "depth": "vrstva (hloubka →)",
        "calls": "volá",
        "lens_label": "logit lens\n„co by řekl teď“", "jlens_label": "J-lens\n„k čemu tlačí později“",
        "tlens_label": "tuned lens\n„opravený readout“",
        "v1": "JISTÝ — od vrstvy 27 mají oba\nreadouty na špici calculator\na už se to nezmění.",
        "v2": "VÁHÁ — uprostřed sahá po „lookup“,\ntoolu, který vůbec nemá, pak přeskakuje\ncalculator ↔ web až do konce.",
        "v3": "ROZMYSLÍ SI TO — ve vrstvě 35 z 36\nJ-lens pořád říká „přijde calculator“ (p ≈ 1,0);\nposlední vrstva to přepíše.",
        "callout1": "vrstvu před výstupem si je\npořád jistý, že odpověď\nje „calculator“",
        "callout2": "poslední vrstva to překlopí",
        "foot1": "logit lens: co by model vypsal, kdyby výpočet skončil v této vrstvě   ·   "
                 "J-lens (Jacobianova lens, Anthropic 2026): co tato vrstva tlačí model říct později",
        "foot2": "Qwen3-4B · greedy, tool_choice: required — každý call vrací identický, validní JSON · "
                 "tohle ilustruje; měřit znamená zasáhnout (steer & rerun) · brainscope — github.com/moudrkat/brainscope",
    }

if LOGIT_ONLY:
    # single-readout story, tuned for a feed: one idea, minimal text
    T["title"] = ("Three tool calls. Same clean JSON. One flipped at the last layer." if not CZ else
                  "Tři tool cally. Stejně čistý JSON. Jeden se překlopil v poslední vrstvě.")
    T["subtitle"] = ("each layer of the model gets to answer early — read at the token "
                     "where the agent writes the tool name" if not CZ else
                     "každá vrstva modelu smí odpovědět předčasně — čteno na tokenu, "
                     "kde agent píše jméno toolu")
    T["v1"] = ("clear question — calculator\non top from layer 25 up" if not CZ else
               "jasná otázka — od vrstvy 25\npořád vede calculator")
    T["v2"] = ("contested — mid-stack it reaches\nfor “lookup”, a tool it doesn't have" if not CZ else
               "sporná — uprostřed sahá po „lookup“,\ntoolu, který nemá")
    T["v3"] = ("calculator wins until two layers\nfrom the end — then it flips" if not CZ else
               "až dvě vrstvy před koncem\nvede calculator — pak přeskočí")
    T["callout1"] = ("two layers from the output,\ncalculator is still winning" if not CZ else
                     "dvě vrstvy před výstupem\npořád vede calculator")
    T["callout2"] = ("the output says web_search" if not CZ else
                     "výstup říká web_search")
    T["foot1"] = ("raw logit lens — an approximate readout: what each layer would answer if generation "
                  "stopped there · Qwen3-4B, greedy · brainscope — github.com/moudrkat/brainscope"
                  if not CZ else
                  "raw logit lens — přibližný readout: co by každá vrstva odpověděla, kdyby generování "
                  "skončilo tam · Qwen3-4B, greedy · brainscope — github.com/moudrkat/brainscope")
    T["foot2"] = ""

if TUNED:
    # the raw lens's known weakness, its trained fix, and an independent
    # second opinion — same verdicts in all three columns
    T["v1"] = ("SURE — from layer 29 up, all three\nreadouts have calculator on top\nand it never changes." if not CZ else
               "JISTÝ — od vrstvy 29 mají všechny tři\nreadouty na špici calculator\na už se to nezmění.")
    T["foot1"] = ("logit lens: raw readout — what each layer would emit   ·   "
                  "tuned lens (Belrose et al. 2023): the same readout with trained per-layer corrections   ·   "
                  "J-lens (Anthropic 2026): what the layer pushes the model to say later"
                  if not CZ else
                  "logit lens: surový readout — co by vrstva vypsala   ·   "
                  "tuned lens (Belrose et al. 2023): týž readout s natrénovanou korekcí po vrstvách   ·   "
                  "J-lens (Anthropic 2026): k čemu vrstva tlačí model později")

Q2 = (["What's the average of", "the boiling points of", "water and ethanol, in °C?"]
      if LOGIT_ONLY else
      ["What's the average of the boiling", "points of water and ethanol, in °C?"])
CASES = [
    ("calc_clear", ["What is 847 × 391?"], "calculator", T["v1"]),
    ("torn_boiling", Q2, "web_search", T["v2"]),
    ("torn_leap_ms", ["How many milliseconds", "are in a leap year?"], "web_search", T["v3"]),
]
COLS = [("lens", T["lens_label"])] if LOGIT_ONLY else \
       [("lens", T["lens_label"]), ("tlens", T["tlens_label"]), ("jlens", T["jlens_label"])] if TUNED else \
       [("lens", T["lens_label"]), ("jlens", T["jlens_label"])]


def classify(tok: str) -> str | None:
    t = tok.strip().lstrip('"').strip().lower()
    if len(t) < 2:
        return None
    if "calculator".startswith(t) or t in ("calculate", "calculator"):
        return "calc"
    if "web_search".startswith(t) or t == "web":
        return "web"
    return None


COL_W, CELL_H = 2.05, 0.46
GAP = 1.6 if LOGIT_ONLY else 0.75   # narrow groups need wider gutters for the headers
GROUP_W = len(COLS) * COL_W + (len(COLS) - 1) * 0.1

fig_w = 3 * GROUP_W + 2 * GAP + 3.2
n_rows = len(LAYERS)
y0 = 2.55                                   # grid bottom
top = y0 + n_rows * CELL_H                  # grid top
fig_h = top + 3.35
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
fig.patch.set_facecolor(PAGE)
ax.set_facecolor(PAGE)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
x0 = 1.35                                   # left margin (layer numbers)

# ---------------- title / subtitle / legend
ax.text(fig_w / 2, fig_h - 0.42, T["title"],
        ha="center", color=INK, fontsize=19, fontweight="bold", family="sans-serif")
ax.text(fig_w / 2, fig_h - 0.86,
        T["subtitle"],
        ha="center", color=INK2, fontsize=12.5, family="sans-serif")

leg = [(C_CALC, T["leg_calc"]), (C_WEB, T["leg_web"]), (SURFACE, T["leg_none"])]
lw_total = sum(0.42 + 0.095 * len(lab) + 0.5 for _, lab in leg)
lx = (fig_w - lw_total) / 2
ly = fig_h - 1.32
for c, lab in leg:
    ax.add_patch(FancyBboxPatch((lx, ly - 0.10), 0.3, 0.21,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc=c, ec=GRID, lw=0.8))
    ax.text(lx + 0.42, ly, lab, ha="left", va="center", color=INK2,
            fontsize=10, family="sans-serif")
    lx += 0.42 + 0.095 * len(lab) + 0.5

# ---------------- layer axis labels
for r, layer in enumerate(LAYERS):
    y = y0 + r * CELL_H
    if (layer + 1) % 2 == 0:
        ax.text(x0 - 0.15, y + CELL_H / 2, f"L{layer + 1}", ha="right", va="center",
                color=MUTED, fontsize=9.5, family="monospace")
ax.text(x0, y0 - 0.30, T["layers_note"], ha="left",
        color=MUTED, fontsize=9, family="sans-serif", style="italic")
ax.text(x0 - 0.75, y0 + n_rows * CELL_H / 2, T["depth"], rotation=90,
        ha="center", va="center", color=MUTED, fontsize=10.5, family="sans-serif")

for gi, (case, qlines, picked, verdict) in enumerate(CASES):
    gx = x0 + gi * (GROUP_W + GAP)
    d = DATA[case]

    # question header (quotes only around the whole thing) + picked chip
    qlines = [("“" if i == 0 else "") + l + ("”" if i == len(qlines) - 1 else "")
              for i, l in enumerate(qlines)]
    qy = top + 1.30 + (len(qlines) - 1) * 0.15
    for li, line in enumerate(qlines):
        ax.text(gx + GROUP_W / 2, qy - li * 0.30, line, ha="center", va="center",
                color=INK, fontsize=(10.5 if LOGIT_ONLY else 11.5), family="sans-serif", fontweight="bold")
    chip_c = C_CALC if picked == "calculator" else C_WEB
    ax.text(gx + GROUP_W / 2, top + 0.68, f"{T['calls']}  {picked}", ha="center", va="center",
            color=chip_c, fontsize=11, family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=chip_c, lw=1.4))

    for ci, (key, label) in enumerate(COLS):
        cx = gx + ci * (COL_W + 0.1)
        ax.text(cx + COL_W / 2, top + 0.24, label, ha="center", va="center",
                color=INK2, fontsize=9.5, family="sans-serif", linespacing=1.25)
        for r, layer in enumerate(LAYERS):
            y = y0 + r * CELL_H
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
                (cx, y + 0.035), COL_W, CELL_H - 0.07,
                boxstyle="round,pad=0,rounding_size=0.06",
                fc=fc, ec=GRID, lw=0.6, alpha=a, mutation_aspect=1))
            tok = e["t"].strip()
            if not all(ord(c) < 0x2500 for c in tok):
                tok = "···"                     # CJK/symbol noise — no glyphs
            if len(tok) > 14:
                tok = tok[:13] + "…"
            ax.text(cx + 0.14, y + CELL_H / 2, tok, ha="left", va="center",
                    color=tc, fontsize=10, family="monospace",
                    fontweight="bold" if tool else "normal")
            if tool and p >= 0.3:
                ax.text(cx + COL_W - 0.12, y + CELL_H / 2, f"{p:.2f}", ha="right",
                        va="center", color=tc, fontsize=8, family="monospace", alpha=0.85)

    # verdict under the group
    ax.text(gx + GROUP_W / 2, y0 - 0.66, verdict, ha="center", va="top",
            color=INK2, fontsize=(9.2 if LOGIT_ONLY else 9.8), family="sans-serif", linespacing=1.45)

# ---------------- callout on torn_leap_ms: J-lens L33–35, or lens L32–L34
gx2 = x0 + 2 * (GROUP_W + GAP)
jx = gx2 + [k for k, _ in COLS].index("jlens" if not LOGIT_ONLY else "lens") * (COL_W + 0.1)
box_y = y0 + ((31 if LOGIT_ONLY else 32) - LAYERS[0]) * CELL_H
ax.add_patch(FancyBboxPatch((jx - 0.05, box_y - 0.02), COL_W + 0.1, 3 * CELL_H + 0.04,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="none", ec=INK, lw=1.8))
ax.annotate(T["callout1"],
            xy=(jx + COL_W + 0.08, box_y + 1.5 * CELL_H),
            xytext=(jx + COL_W + 0.45, box_y + 0.9 * CELL_H),
            color=INK, fontsize=10, family="sans-serif", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.2), va="center")
ax.annotate(T["callout2"],
            xy=(jx + COL_W + 0.08, y0 + (35 - LAYERS[0]) * CELL_H + CELL_H / 2),
            xytext=(jx + COL_W + 0.45, y0 + (35 - LAYERS[0]) * CELL_H + CELL_H / 2),
            color=C_WEB, fontsize=10, family="sans-serif", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C_WEB, lw=1.2), va="center")

# ---------------- footer
ax.text(x0 - 0.9, 0.62,
        T["foot1"],
        color=MUTED, fontsize=9.5, family="sans-serif")
ax.text(x0 - 0.9, 0.30,
        T["foot2"],
        color=MUTED, fontsize=9.5, family="sans-serif")

name = "fig_hesitation_" + ("logit_" if LOGIT_ONLY else "tuned_" if TUNED else "") + ("cz" if CZ else "en")
fig.savefig(f"{SCRATCH}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.25)
print("saved", name)
