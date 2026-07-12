"""The readout river: at every layer, 100% of cases split by which token is
the tuned-lens top-1 — each token is one band of the river. Early layers:
unstructured leftovers. Mid-stack: capability tokens. Final layers: the
schema name absorbs the flow. Top-1 shares sum to 1 at every layer, so the
geometry is exact, not illustrative.

uv run --with matplotlib --no-project python3 fig/render_vocab_river.py [cz]

`groups` mode renders one river per question group instead (4 panels);
on the torn groups the river visibly forks into both schema names:
uv run --with matplotlib --no-project python3 fig/render_vocab_river.py groups [cz]
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
C_CALC, C_WEB = "#c98500", "#3987e5"
TEALS = ["#2fa889", "#57c3a5", "#1f7a63", "#7fd9bf", "#175c4b"]
GRAYS = ["#3a4358", "#2c3448", "#465071", "#252d40", "#525d80", "#1f2636"]

GENERIC = {"search", "lookup", "look", "query", "google", "api", "database",
           "find", "fetch", "browse", "retrieve", "internet", "calculate",
           "calc", "calculation", "compute", "computing", "math", "evaluate",
           "formula", "conversion", "function"}
SCHEMA = {"web", "web_search", "calculator"}

CZ = "cz" in sys.argv[1:]
BY_GROUP = "groups" in sys.argv[1:]
GROUP_NAMES = ("clear_calc", "clear_search", "torn_const", "torn_fact")
ONE_GROUP = next((a for a in sys.argv[1:] if a in GROUP_NAMES), None)
T = {
    "title": "The readout river — top-1 token composition by depth",
    "sub": "at each layer, all cases split by which token tops the tuned-lens readout at the tool-name token · "
           "%d questions · exact geometry: shares sum to 100%% at every layer",
    "web": "cases that called web_search (n=%d)",
    "calc": "cases that called calculator (n=%d)",
    "layer": "layer (depth →)",
    "share": "share of cases",
    "other": "other tokens",
    "leg": "band color: blue/orange = schema name · teal = capability tokens · gray = unstructured readouts",
    "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023), exact readouts from stored hidden states · "
            "battery: vocab.py · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Řeka readoutů — složení top-1 tokenů po hloubce",
        "sub": "v každé vrstvě se všechny cases rozdělí podle toho, který token vede tuned-lens readout na tokenu jména toolu · "
               "%d otázek · exaktní geometrie: podíly dávají v každé vrstvě 100 %%",
        "web": "cases volající web_search (n=%d)",
        "calc": "cases volající calculator (n=%d)",
        "layer": "vrstva (hloubka →)",
        "share": "podíl cases",
        "other": "ostatní tokeny",
        "leg": "barva pruhu: modrá/oranžová = jméno ze schématu · tyrkys = capability tokeny · šedá = nestrukturované readouty",
        "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023), exaktní readouty z uložených hidden states · "
                "battery: vocab.py · brainscope — github.com/moudrkat/brainscope",
    }

N_BANDS = 11


def river(cases):
    counts = defaultdict(lambda: [0] * 36)
    n = len(cases)
    for v in cases.values():
        for l in range(36):
            t = v["top8"]["tuned"][l][0]["t"].strip().lstrip('"').strip().lower()
            counts[t if len(t) >= 2 else "·"][l] += 1
    keep = sorted(counts, key=lambda t: -sum(counts[t]))[:N_BANDS]
    keep.sort(key=lambda t: max(range(36), key=lambda l: counts[t][l]))
    bands = {t: [c / n for c in counts[t]] for t in keep}
    other = [1 - sum(bands[t][l] for t in keep) for l in range(36)]
    return bands, other


def color_for(tok, gi, ti):
    if tok in ("calculator",):
        return C_CALC
    if tok in ("web", "web_search"):
        return C_WEB
    if tok in GENERIC:
        c = TEALS[ti[0] % len(TEALS)]; ti[0] += 1
        return c
    c = GRAYS[gi[0] % len(GRAYS)]; gi[0] += 1
    return c


GROUP_TITLES = {
    "clear_calc": ("clear arithmetic (n=%d)", "čistá aritmetika (n=%d)"),
    "clear_search": ("clear current facts (n=%d)", "čistá aktuální fakta (n=%d)"),
    "torn_const": ("torn: constant + math (n=%d)", "torn: konstanta + výpočet (n=%d)"),
    "torn_fact": ("torn: fact + math (n=%d)", "torn: fakt + výpočet (n=%d)"),
}
if ONE_GROUP:
    PANELS = [({k: v for k, v in DATA.items() if v["group"] == ONE_GROUP},
               GROUP_TITLES[ONE_GROUP][1 if CZ else 0])]
elif BY_GROUP:
    PANELS = [({k: v for k, v in DATA.items() if v["group"] == g},
               GROUP_TITLES[g][1 if CZ else 0])
              for g in GROUP_NAMES]
else:
    PANELS = [({k: v for k, v in DATA.items() if v["picked"] == "web_search"}, T["web"]),
              ({k: v for k, v in DATA.items() if v["picked"] == "calculator"}, T["calc"])]

fig, axes = plt.subplots(len(PANELS), 1,
                         figsize=(13.2, 4.5 * len(PANELS) + 0.8), dpi=160,
                         gridspec_kw={"hspace": 0.30})
if len(PANELS) == 1:
    axes = [axes]
fig_h = 4.5 * len(PANELS) + 0.8
fig.subplots_adjust(top=1 - 1.25 / fig_h, bottom=0.5 / fig_h + 0.04)
fig.patch.set_facecolor(PAGE)
fig.suptitle(T["title"], color=INK, fontsize=17, fontweight="bold",
             family="sans-serif", y=1 - 0.22 / fig_h)
fig.text(0.5, 1 - 0.58 / fig_h, T["sub"] % len(DATA), ha="center", color=INK2,
         fontsize=10, family="sans-serif")
fig.text(0.5, 1 - 0.82 / fig_h, T["leg"], ha="center", color=MUTED, fontsize=8.8,
         family="sans-serif")

for ax, (cases, ptitle) in zip(axes, PANELS):
    bands, other = river(cases)
    gi, ti = [0], [0]
    toks = list(bands)
    colors = [color_for(t, gi, ti) for t in toks]
    xs = list(range(1, 37))
    ys = [bands[t] for t in toks] + [other]
    ax.stackplot(xs, ys, colors=colors + [PAGE], lw=0.7,
                 edgecolor=PAGE, baseline="zero")
    ax.set_facecolor(PAGE)
    ax.set_xlim(1, 36)
    ax.set_ylim(0, 1)
    ax.set_title(ptitle % len(cases), color=INK2, fontsize=11,
                 family="sans-serif", loc="left", pad=8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_ylabel(T["share"], color=MUTED, fontsize=9.5)
    # inline label at each band's widest point
    cum = [0.0] * 36
    for t, ys_t, c in zip(toks + [T["other"]], ys, colors + [PAGE]):
        widths = ys_t
        l = max(range(36), key=lambda i: widths[i])
        if widths[l] >= 0.055:
            y_mid = cum[l] + widths[l] / 2
            big = widths[l] >= 0.30
            ax.text(l + 1, y_mid, t, ha="center", va="center",
                    color=INK if big else INK2,
                    fontsize=11 if big else 8.5, family="monospace",
                    fontweight="bold" if big else "normal")
        cum = [cum[i] + widths[i] for i in range(36)]
axes[-1].set_xlabel(T["layer"], color=MUTED, fontsize=9.5)
fig.text(0.01, 0.005, T["foot"], color=MUTED, fontsize=8, family="sans-serif")

name = "fig_vocab_river_" + (f"{ONE_GROUP}_" if ONE_GROUP else
                             "groups_" if BY_GROUP else "") + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.3)
print("saved", name)
