"""The word race: tokens written straight into the chart, sized by their
probability peak, with a thin line tracing each curve across depth.

Modes (combine freely):
  (default)            two aggregate panels (web-bound / calculator-bound)
  clear_calc | clear_search | torn_const | torn_fact
                       one panel, that question group only
  case:<case_id>       one panel, ONE question (exact per-case probabilities,
                       prompt shown) — e.g. case:torn_const_b229
  light                white background (palette validated for light surface)
  cz                   Czech labels

uv run --with matplotlib --no-project python3 fig/render_wordrace.py [modes...]
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

CZ = "cz" in sys.argv[1:]
LIGHT = "light" in sys.argv[1:]
GROUP_NAMES = ("clear_calc", "clear_search", "torn_const", "torn_fact")
ONE_GROUP = next((a for a in sys.argv[1:] if a in GROUP_NAMES), None)
ONE_CASE = next((a.split(":", 1)[1] for a in sys.argv[1:]
                 if a.startswith("case:")), None)

if LIGHT:
    PAGE, SURFACE = "#ffffff", "#f2f4f8"
    INK, INK2, MUTED = "#1a2233", "#3c4658", "#6a7590"
    GRID = "#d8dde8"
    C_CALC, C_WEB, C_GEN = "#9c6a00", "#2f6fc4", "#1f8a6d"
    GRAY = "#8a93a8"
else:
    PAGE, SURFACE = "#0c1322", "#111a2e"
    INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
    GRID = "#232c42"
    C_CALC, C_WEB, C_GEN = "#c98500", "#3987e5", "#2fa889"
    GRAY = "#6a7590"

T = {
    "title": "The word race — what the readout says on the way to the tool name",
    "title_case": "One tool call, watched from the inside",
    "sub": "each word sits at the decoder layer where its mean tuned-lens probability peaks, sized by that peak · "
           "%d questions · top-8 readouts (a lower bound)",
    "sub_case": "each word = a token in the tuned-lens readout at the token where the agent writes the tool name ·\n"
                "word position & line = its probability across depth, word size = its peak",
    "web": "cases that called web_search (n=%d)",
    "calc": "cases that called calculator (n=%d)",
    "xlabel": "decoder layer ℓ  (computation depth →, 36 layers total)",
    "ylabel": "mean tuned-lens probability  p̄(token)",
    "ylabel_case": "tuned-lens probability  p(token)",
    "ann_mid": "mid-stack, the readout reaches\nfor computing: “calculate”, then\n“calculator” — the other racer",
    "ann_end": "the last two layers flip it —\n“web” wins and web_search\nis what gets called",
    "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023): logit lens with trained per-layer corrections · "
            "battery: vocab.py · brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Slovní závod — co říká readout cestou ke jménu toolu",
        "title_case": "Jeden tool call, sledovaný zevnitř",
        "sub": "každé slovo sedí ve vrstvě, kde vrcholí jeho průměrné tuned-lens p, velikost = ten vrchol · "
               "%d otázek · top-8 readouty (dolní odhad)",
        "sub_case": "každé slovo = token v tuned-lens readoutu na tokenu, kde agent píše jméno toolu ·\n"
                    "pozice slova a čára = jeho pravděpodobnost po hloubce, velikost slova = vrchol",
        "web": "cases volající web_search (n=%d)",
        "calc": "cases volající calculator (n=%d)",
        "xlabel": "vrstva dekodéru ℓ  (hloubka výpočtu →, celkem 36 vrstev)",
        "ylabel": "průměrná tuned-lens pravděpodobnost  p̄(token)",
        "ylabel_case": "tuned-lens pravděpodobnost  p(token)",
        "ann_mid": "uprostřed sítě readout sahá\npo počítání: „calculate“, pak\n„calculator“ — druhý závodník",
        "ann_end": "poslední dvě vrstvy to překlopí —\nvyhraje „web“ a volá se\nweb_search",
        "foot": "Qwen3-4B · tuned lens (Belrose et al. 2023): logit lens s natrénovanou korekcí po vrstvách · "
                "battery: vocab.py · brainscope — github.com/moudrkat/brainscope",
    }

GENERIC = {"search", "lookup", "look", "query", "google", "api", "database",
           "find", "fetch", "browse", "retrieve", "internet", "calculate",
           "calc", "calculation", "compute", "computing", "math", "evaluate",
           "formula", "conversion", "function"}
GROUP_TITLES = {
    "clear_calc": ("clear arithmetic", "čistá aritmetika"),
    "clear_search": ("clear current facts", "čistá aktuální fakta"),
    "torn_const": ("torn: constant + math", "torn: konstanta + výpočet"),
    "torn_fact": ("torn: fact + math", "torn: fakt + výpočet"),
}


def case_prompt(case_id):
    """census stores no prompts; the battery is deterministic, regenerate"""
    sys.path.insert(0, os.path.dirname(HERE))
    import vocab
    groups = dict(vocab.big_groups()) if "_b" in case_id else dict(vocab.GROUPS)
    grp, idx = case_id.rsplit("_", 1)
    return groups[grp][int(idx.lstrip("b"))]


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


def draw_panel(ax, cs, ptitle, ylabel, annotate_flip=False):
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 36])
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(1, 36.8)
    ax.set_ylim(0, 1.06)
    ax.set_title(ptitle, color=INK2, fontsize=11, family="sans-serif",
                 loc="left", pad=8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    placed = []
    for t, c in sorted(cs.items(), key=lambda kv: max(kv[1])):
        color = tok_color(t)
        ax.plot(range(1, 37), c, color=color, lw=1.2, alpha=0.5)
        pl = max(range(36), key=lambda l: c[l])
        px, py = pl + 1, max(c)
        size = 8 + 26 * min(1.0, py / 0.9)
        for qx, qy in placed:
            if abs(px - qx) < 2.6 and abs(py - qy) < 0.055:
                py += 0.055
        placed.append((px, py))
        ax.text(min(px, 35.6), min(py + 0.015, 0.99), t, ha="center",
                va="bottom", color=color, fontsize=size, family="monospace",
                fontweight="bold" if py > 0.10 else "normal")
    if annotate_flip and "web" in cs:
        mid = cs.get("calculate") or cs.get("calculator")
        if mid:
            pl = max(range(36), key=lambda l: mid[l])
            ax.annotate(T["ann_mid"], xy=(pl + 0.7, mid[pl]),
                        xytext=(8.5, 0.74),
                        color=C_GEN if "calculate" in cs else C_CALC,
                        fontsize=10.5, family="sans-serif",
                        linespacing=1.45, ha="center",
                        arrowprops=dict(arrowstyle="->", color=C_GEN if
                                        "calculate" in cs else C_CALC,
                                        lw=1.3, shrinkB=10))
        cw = cs["web"]
        ax.annotate(T["ann_end"], xy=(35.9, cw[35] - 0.06),
                    xytext=(10.5, 0.38),
                    color=C_WEB, fontsize=10.5, family="sans-serif",
                    linespacing=1.45, ha="center",
                    arrowprops=dict(arrowstyle="->", color=C_WEB, lw=1.3,
                                    shrinkB=12))


if ONE_CASE:
    v = DATA[ONE_CASE]
    cs = curves({ONE_CASE: v}, min_peak=0.05)
    prompt = case_prompt(ONE_CASE)
    fig, ax = plt.subplots(figsize=(12.0, 6.4), dpi=160)
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(top=0.76, bottom=0.11, left=0.08, right=0.97)
    fig.suptitle(T["title_case"], color=INK, fontsize=17.5, fontweight="bold",
                 family="sans-serif", y=0.97)
    fig.text(0.5, 0.885, f"“{prompt}”", ha="center", color=INK,
             fontsize=12.5, family="monospace", fontweight="bold")
    fig.text(0.5, 0.80, T["sub_case"], ha="center", color=INK2,
             fontsize=9.5, family="sans-serif", linespacing=1.4)
    picked = "web_search" if v["picked"] == "web_search" else "calculator"
    draw_panel(ax, cs, f"→ {('volá' if CZ else 'calls')} {picked}",
               T["ylabel_case"], annotate_flip=(v["picked"] == "web_search"))
    ax.set_xlabel(T["xlabel"], color=INK2, fontsize=10)
    fig.text(0.01, 0.008, T["foot"], color=MUTED, fontsize=8, family="sans-serif")
    name = f"fig_wordrace_{ONE_CASE}_" + ("light_" if LIGHT else "") + ("cz" if CZ else "en")
elif ONE_GROUP:
    cases = {k: v for k, v in DATA.items() if v["group"] == ONE_GROUP}
    gt = GROUP_TITLES[ONE_GROUP][1 if CZ else 0]
    fig, ax = plt.subplots(figsize=(12.6, 6.2), dpi=160)
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.08, right=0.97)
    fig.suptitle(T["title"], color=INK, fontsize=16, fontweight="bold",
                 family="sans-serif", y=0.965)
    fig.text(0.5, 0.855, T["sub"] % len(cases), ha="center", color=INK2,
             fontsize=9.8, family="sans-serif")
    draw_panel(ax, curves(cases), f"{gt} (n={len(cases)})", T["ylabel"])
    ax.set_xlabel(T["xlabel"], color=INK2, fontsize=10)
    fig.text(0.01, 0.008, T["foot"], color=MUTED, fontsize=8, family="sans-serif")
    name = f"fig_wordrace_{ONE_GROUP}_" + ("light_" if LIGHT else "") + ("cz" if CZ else "en")
else:
    web = {k: v for k, v in DATA.items() if v["picked"] == "web_search"}
    calc = {k: v for k, v in DATA.items() if v["picked"] == "calculator"}
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 10.6), dpi=160,
                             gridspec_kw={"hspace": 0.32})
    fig.patch.set_facecolor(PAGE)
    fig.suptitle(T["title"], color=INK, fontsize=17, fontweight="bold",
                 family="sans-serif", y=0.975)
    fig.text(0.5, 0.925, T["sub"] % len(DATA), ha="center", color=INK2,
             fontsize=9.8, family="sans-serif")
    draw_panel(axes[0], curves(web), T["web"] % len(web), T["ylabel"])
    draw_panel(axes[1], curves(calc), T["calc"] % len(calc), T["ylabel"])
    axes[1].set_xlabel(T["xlabel"], color=INK2, fontsize=10)
    fig.text(0.01, 0.005, T["foot"], color=MUTED, fontsize=8.2, family="sans-serif")
    name = "fig_wordrace_" + ("light_" if LIGHT else "") + ("cz" if CZ else "en")

fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight", pad_inches=0.3)
print("saved", name)
