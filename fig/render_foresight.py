"""Foresight figure: the sentence the agent wrote, and what each lens saw
under every word of it — the banned concept held toward the future call.

deps: matplotlib only — run e.g.
    uv run --with matplotlib --no-project python3 fig/render_foresight.py [cz]
after `python3 foresight.py --dump fig/foresight.json`.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{HERE}/foresight.json"))

SURFACE, PAGE = "#111a2e", "#0c1322"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#8a8f9e"
GRID = "#232c42"
C_CALC, C_WEB = "#c98500", "#3987e5"

CZ = "cz" in sys.argv[1:]
T = {
    "title": "The prompt bans the word “calculate”. The J-lens shows the model holding it anyway.",
    "subtitle": "One neutral sentence first (tool-ish words forbidden), then a tool call — exact p of the calculate/calculator word family under every token of the sentence",
    "lens": "logit lens — “next word”", "jlens": "J-lens — “pushed toward later”",
    "call": "→ tool call",
    "calls": "calls",
    "v_calc": "at the verb slot — where “calculate” would go — the banned word shows up:\na suppressed next-word candidate in the logit lens, and 4× stronger in the J-lens,\nheld toward the calculator call written 15 tokens later",
    "v_torn": "same verb slot, same held concept — the call comes 11 tokens later",
    "v_search": "control: the same probe on a search question stays dark — the mass above\nis the calculation concept surfacing, not generic tool-call noise",
    "foot1": "J-lens (Jacobian lens, Anthropic 2026): what each layer pushes the model to say LATER · exact readouts from stored hidden "
             "states, best layer per step, p summed over the word family (calculator, calculate, calculation, calc, …) · shade saturates at p = 0.5",
    "foot2": "Qwen3-4B · greedy · the sentence may not contain tool names or verbs like calculate/search, so the logit lens (next word) "
             "should see nothing · illustrates, not measures — the intervention (steer & rerun) is the next step · "
             "brainscope — github.com/moudrkat/brainscope",
}
if CZ:
    T = {
        "title": "Prompt zakazuje slovo „calculate“. J-lens ukazuje, že ho model stejně drží.",
        "subtitle": "Napřed jedna neutrální věta (slova o toolech zakázána), pak tool call — exaktní p rodiny slov calculate/calculator pod každým tokenem věty",
        "lens": "logit lens — „další slovo“", "jlens": "J-lens — „k čemu tlačí později“",
        "call": "→ tool call",
        "calls": "volá",
        "v_calc": "na pozici slovesa — kam by patřilo „calculate“ — zakázané slovo prosvítá:\npotlačený kandidát v logit lens a 4× silněji v J-lens,\ndržený směrem ke calculator callu napsanému o 15 tokenů později",
        "v_torn": "stejná pozice slovesa, stejný držený koncept — call přijde o 11 tokenů později",
        "v_search": "kontrola: stejná sonda na search otázce zůstává temná — masa nahoře\nje vynořující se koncept počítání, ne obecný šum tool callů",
        "foot1": "J-lens (Jacobianova lens, Anthropic 2026): co každá vrstva tlačí model říct POZDĚJI · exaktní readouty z uložených hidden "
                 "states, nejlepší vrstva na krok, p sečteno přes rodinu slov (calculator, calculate, calculation, calc, …) · sytost saturuje na p = 0,5",
        "foot2": "Qwen3-4B · greedy · věta nesmí obsahovat jména toolů ani slovesa calculate/search, takže logit lens (další slovo) "
                 "nemá co vidět · ilustruje, neměří — další krok je intervence (steer & rerun) · "
                 "brainscope — github.com/moudrkat/brainscope",
    }

ORDER = [("fore_calc", T["v_calc"]), ("fore_torn", T["v_torn"]),
         ("fore_search", T["v_search"])]
ORDER = [(k, v) for k, v in ORDER if DATA.get(k)]
CELL_W, CELL_H, ROW_GAP = 0.62, 0.42, 0.10
x0 = 3.1
fig_w = x0 + max(len(DATA[k]["steps"]) for k, _ in ORDER) * CELL_W + 2.1

blocks, y_cursor = [], 1.15
for k, v in reversed(ORDER):
    verdict_lines = v.count("\n") + 1
    blocks.append((k, v, y_cursor + verdict_lines * 0.26 + 0.18))
    y_cursor += 2 * CELL_H + ROW_GAP + 1.15 + verdict_lines * 0.26 + 0.42
fig_h = y_cursor + 1.55
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
fig.patch.set_facecolor(PAGE)
ax.set_facecolor(PAGE)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

ax.text(fig_w / 2, fig_h - 0.42, T["title"], ha="center", color=INK,
        fontsize=16.5, fontweight="bold", family="sans-serif")
ax.text(fig_w / 2, fig_h - 0.80, T["subtitle"], ha="center", color=INK2,
        fontsize=10, family="sans-serif")

for k, verdict, yb in blocks:
    d = DATA[k]
    color = C_CALC if d["picked"] == "calculator" else C_WEB
    steps = d["steps"]
    for ri, (key, label) in enumerate([("jlens", T["jlens"]), ("lens", T["lens"])]):
        y = yb + ri * (CELL_H + ROW_GAP)
        ax.text(x0 - 0.18, y + CELL_H / 2, label, ha="right", va="center",
                color=INK2 if key == "jlens" else MUTED, fontsize=9.5,
                family="sans-serif",
                fontweight="bold" if key == "jlens" else "normal")
        for si, s in enumerate(steps):
            p = s[key]
            cx = x0 + si * CELL_W
            a = 0.14 + 0.86 * min(1.0, p / 0.5) if p > 0.02 else 1.0
            ax.add_patch(FancyBboxPatch(
                (cx, y), CELL_W - 0.07, CELL_H,
                boxstyle="round,pad=0,rounding_size=0.05",
                fc=C_CALC if p > 0.02 else SURFACE, ec=GRID, lw=0.6, alpha=a))
            if p >= 0.05:
                ax.text(cx + (CELL_W - 0.07) / 2, y + CELL_H / 2, f"{p:.2f}"[1:],
                        ha="center", va="center", color=INK, fontsize=7.5,
                        family="monospace")
    yt = yb + 2 * (CELL_H + ROW_GAP) + 0.06
    for si, s in enumerate(steps):
        ax.text(x0 + si * CELL_W + (CELL_W - 0.07) / 2, yt, s["tok"].strip() or "␣",
                ha="center", va="bottom", color=INK, fontsize=8.5,
                family="monospace", rotation=45)
    cx_end = x0 + len(steps) * CELL_W + 0.05
    ax.text(cx_end, yb + CELL_H + ROW_GAP / 2, f"{T['call']}", ha="left",
            va="center", color=color, fontsize=9.5, family="sans-serif",
            fontweight="bold")
    ax.text(x0 - 2.95, yt + 0.12,
            f"{T['calls']} {d['picked']}", ha="left", va="bottom",
            color=color, fontsize=10, family="monospace", fontweight="bold")
    ax.text(x0, yb - 0.22, verdict, ha="left", va="top", color=INK2,
            fontsize=9.2, family="sans-serif", linespacing=1.45)

ax.text(0.35, 0.60, T["foot1"], color=MUTED, fontsize=8.6, family="sans-serif")
ax.text(0.35, 0.30, T["foot2"], color=MUTED, fontsize=8.6, family="sans-serif")

name = "fig_foresight_" + ("cz" if CZ else "en")
fig.savefig(f"{HERE}/{name}.png", facecolor=PAGE, bbox_inches="tight",
            pad_inches=0.25)
print("saved", name)
