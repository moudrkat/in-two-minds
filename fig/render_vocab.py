"""Vocabulary census figures from vocab_census.json (fig/vocab_extract.py).

Fig A (handoff): mean exact p of concept families vs layer, tuned lens,
split by the tool the case finally called. Fig B (cascade): which literal
tokens are top-1 at which depth, as a share of cases — the words the model
thinks in before the schema name.

uv run --with matplotlib --no-project python3 fig/render_vocab.py [cz]
"""
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{HERE}/vocab_census.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_CALC, C_WEB, C_GEN = "#c98500", "#3987e5", "#2fa889"   # validated triple

CZ = "cz" in sys.argv[1:]
T = {
    "handoff_title": "The handoff: capability words first, the schema name last",
    "handoff_sub": "mean exact tuned-lens probability at the tool-name token, %d questions · "
                   "the model thinks in generic verbs mid-stack and snaps to the tool's literal name late",
    "web_panel": "cases that called web_search (n=%d)",
    "calc_panel": "cases that called calculator (n=%d)",
    "s_schema_web": "“web_search” (schema name)",
    "s_generic_web": "retrieval words (search, lookup,\nquery, google, api, …)",
    "s_rival_calc": "“calculator” (rival schema)",
    "s_schema_calc": "“calculator” (schema name)",
    "s_generic_calc": "compute words (calculate,\ncompute, math, …)",
    "s_rival_web": "“web_search” (rival schema)",
    "layer": "layer (depth →)",
    "meanp": "mean p (exact, tuned lens)",
    "casc_title": "The words in the middle — what the model calls its tools before it knows their names",
    "casc_sub": "share of cases where the token is tuned-lens top-1 at that layer · rows sorted by where they peak · %d questions",
    "casc_web": "cases that called web_search",
    "casc_calc": "cases that called calculator",
    "share": "share of cases",
    "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023) readouts from stored hidden states, full softmax (no top-k truncation) · "
            "battery: vocab.py (30 clear-calc, 30 clear-search, 30 torn constant+math, 30 torn fact+math) · "
            "brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "handoff_title": "Předávka: napřed slova o schopnostech, jméno toolu až nakonec",
        "handoff_sub": "průměrné exaktní p (tuned lens) na tokenu jména toolu, %d otázek · "
                       "uprostřed sítě model myslí v obecných slovesech, na doslovné jméno přepne pozdě",
        "web_panel": "cases volající web_search (n=%d)",
        "calc_panel": "cases volající calculator (n=%d)",
        "s_schema_web": "„web_search“ (jméno ze schématu)",
        "s_generic_web": "slova o hledání (search, lookup,\nquery, google, api, …)",
        "s_rival_calc": "„calculator“ (rival)",
        "s_schema_calc": "„calculator“ (jméno ze schématu)",
        "s_generic_calc": "slova o počítání (calculate,\ncompute, math, …)",
        "s_rival_web": "„web_search“ (rival)",
        "layer": "vrstva (hloubka →)",
        "meanp": "průměrné p (exaktní, tuned lens)",
        "casc_title": "Slova uprostřed — jak model říká svým toolům, než zná jejich jména",
        "casc_sub": "podíl cases, kde je token tuned-lens top-1 v dané vrstvě · řádky seřazené podle vrcholu · %d otázek",
        "casc_web": "cases volající web_search",
        "casc_calc": "cases volající calculator",
        "share": "podíl cases",
        "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023) readouty z uložených hidden states, plný softmax (žádné top-k ořezy) · "
                "battery: vocab.py (30 clear-calc, 30 clear-search, 30 torn konstanta+výpočet, 30 torn fakt+výpočet) · "
                "brainscope — github.com/moudrkat/brainscope",
    }

web_cases = {k: v for k, v in DATA.items() if v["picked"] == "web_search"}
calc_cases = {k: v for k, v in DATA.items() if v["picked"] == "calculator"}
LAYERS = list(range(36))


def fam_mean(cases, fam):
    return [sum(v["fam"]["tuned"][fam][l] for v in cases.values()) / max(1, len(cases))
            for l in LAYERS]


# ---------------- Fig A: handoff curves
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=160, sharey=True)
fig.patch.set_facecolor(PAGE)
fig.suptitle(T["handoff_title"], color=INK, fontsize=15.5, fontweight="bold",
             family="sans-serif", y=1.00)
fig.text(0.5, 0.925, T["handoff_sub"] % len(DATA), ha="center", color=INK2,
         fontsize=9.5, family="sans-serif")

PANELS = [
    (axes[0], web_cases, T["web_panel"],
     [("schema_web", C_WEB, T["s_schema_web"], "-"),
      ("generic_web", C_GEN, T["s_generic_web"], "-"),
      ("schema_calc", C_CALC, T["s_rival_calc"], "--")]),
    (axes[1], calc_cases, T["calc_panel"],
     [("schema_calc", C_CALC, T["s_schema_calc"], "-"),
      ("generic_calc", C_GEN, T["s_generic_calc"], "-"),
      ("schema_web", C_WEB, T["s_rival_web"], "--")]),
]
for ax, cases, ptitle, series in PANELS:
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(ptitle % len(cases), color=INK2, fontsize=10.5,
                 family="sans-serif", pad=8)
    ax.set_xlabel(T["layer"], color=MUTED, fontsize=9.5)
    ax.set_xlim(1, 36)
    ax.set_ylim(0, 1.0)
    for fam, color, label, style in series:
        ys = fam_mean(cases, fam)
        ax.plot(range(1, 37), ys, style, color=color, lw=2, label=label)
        peak_l = max(range(36), key=lambda l: ys[l])
        end_y = ys[-1]
        ax.annotate(label.split("\n")[0], xy=(36, end_y),
                    xytext=(36.4, end_y + (0.03 if fam.startswith("schema") else -0.02)),
                    color=color, fontsize=8, family="sans-serif",
                    va="center", annotation_clip=False)
axes[0].set_ylabel(T["meanp"], color=MUTED, fontsize=9.5)
fig.text(0.01, -0.04, T["foot"], color=MUTED, fontsize=7.8, family="sans-serif")
fig.tight_layout(rect=(0, 0, 0.93, 0.9))
name = "fig_vocab_handoff_" + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.3)
print("saved", name)

# ---------------- Fig B: cascade heatmaps
def cascade(cases):
    """rows: token form (stripped, lowered) -> [share of cases top-1 at layer l]"""
    counts = defaultdict(lambda: [0] * 36)
    for v in cases.values():
        for l in LAYERS:
            t = v["top8"]["tuned"][l][0]["t"].strip().lstrip('"').strip().lower()
            if len(t) >= 2:
                counts[t][l] += 1
    n = max(1, len(cases))
    totals = {t: sum(c) for t, c in counts.items()}
    keep = sorted((t for t, tot in totals.items() if tot >= max(2, 0.03 * n * 4)),
                  key=lambda t: -totals[t])[:14]
    keep.sort(key=lambda t: max(range(36), key=lambda l: counts[t][l]))
    return {t: [c / n for c in counts[t]] for t in keep}


ramp_web = LinearSegmentedColormap.from_list("w", [PAGE, "#2b66ac", "#7db8f5"])
ramp_calc = LinearSegmentedColormap.from_list("c", [PAGE, "#9a6a08", "#f5c35e"])

casc_w, casc_c = cascade(web_cases), cascade(calc_cases)
hw, hc = len(casc_w), len(casc_c)
fig_h = 2.3 + (hw + hc) * 0.34 + 1.6
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.4, fig_h), dpi=160,
                             gridspec_kw={"height_ratios": [max(hw, 1), max(hc, 1)],
                                          "hspace": 0.35})
fig.patch.set_facecolor(PAGE)
fig.suptitle(T["casc_title"], color=INK, fontsize=15, fontweight="bold",
             family="sans-serif", y=0.99)
fig.text(0.5, 0.955, T["casc_sub"] % len(DATA), ha="center", color=INK2,
         fontsize=9.5, family="sans-serif")

for ax, casc, ramp, ptitle in [(a1, casc_w, ramp_web, T["casc_web"]),
                               (a2, casc_c, ramp_calc, T["casc_calc"])]:
    ax.set_facecolor(PAGE)
    toks = list(casc)
    mat = [casc[t] for t in toks]
    im = ax.imshow(mat, aspect="auto", cmap=ramp, vmin=0, vmax=1,
                   extent=(0.5, 36.5, len(toks) - 0.5, -0.5))
    ax.set_yticks(range(len(toks)))
    ax.set_yticklabels(toks, color=INK2, fontsize=9, family="monospace")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 36])
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(ptitle, color=INK2, fontsize=10.5, family="sans-serif",
                 loc="left", pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(colors=MUTED, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)
    cb.set_label(T["share"], color=MUTED, fontsize=8)
a2.set_xlabel(T["layer"], color=MUTED, fontsize=9.5)
fig.text(0.01, 0.01, T["foot"], color=MUTED, fontsize=7.8, family="sans-serif")
name = "fig_vocab_cascade_" + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.3)
print("saved", name)
