"""Read the agent's mind after the fact: was the tool choice ever contested?

Pulls the traces that agent.py just produced from brainscope's trace API and
zooms in on ONE generated token per trace: the first token of the tool name
inside the tool call. At that position the stored logit-lens readout says,
for every layer, what the model would emit if it stopped there — so we can
watch the decision crystallize with depth, and see whether the OTHER tool
was winning in the middle of the stack.

Reported per case:

  layers   one char per decoder layer at the choice token:
           C = calculator on top, W = web_search on top, . = something else
  settle   first layer from which the winner stays on top to the end —
           deeper settle = the model stayed torn for longer
  margin   p(picked) - p(rival) at the last layer's readout, i.e. (up to
           readout precision) in the distribution the token was sampled
           from — near zero = the call was a coin flip
  rival    the losing tool's probability: its peak (and layer) at the choice
           token, and whether it surfaced anywhere earlier in the generation

Numbers from stored top-5 readouts are a LOWER BOUND (a tool absent from a
layer's top-5 counts as 0 there). This illustrates hesitation on your model
and your prompts; measuring it properly means interventions and controls.

Usage: python hesitation.py [--url http://localhost:8010]
"""

import argparse
import json
import re
import urllib.request

TOOL_NAMES = ["calculator", "web_search"]   # override with --tools a,b


def get(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def classify(entry_token: str) -> str | None:
    """Map a lens top-5 token to a tool name by prefix (>=2 chars),
    case-insensitively — mid-stack the model often reads 'Calculator'."""
    t = entry_token.strip().lstrip('"').strip().lower()
    for name in TOOL_NAMES:
        if len(t) >= 2 and name.lower().startswith(t):
            return name
    return None


def choice_step(all_tokens: list[str]) -> int | None:
    """Index of the generated token that starts the tool name value.

    With tool_choice forcing, brainscope seeds `<tool_call>\n{"name": ` as
    part of the PROMPT, so the generated tokens open mid-call and the
    `"name":` key never appears here — then the choice token is simply the
    first generated token that reads as a tool name."""
    text = "".join(all_tokens)
    m = re.search(r'"name"\s*:\s*"', text)
    if m:
        pos, acc = m.end(), 0
        for i, tok in enumerate(all_tokens):
            acc += len(tok)
            if acc > pos:
                return i
        return None
    for i, tok in enumerate(all_tokens):
        if classify(tok):
            return i
    return None


def analyze(trace: dict, key: str = "lens") -> dict | None:
    step = choice_step(trace["all_tokens"])
    if step is None:
        return None
    lens_idx = step - trace.get("capture_offset", 0)
    lens = trace.get(key) or []
    if not (0 <= lens_idx < len(lens)) or lens[lens_idx] is None:
        return None

    picked = classify(trace["all_tokens"][step])
    if picked is None:   # unusual tokenization; bail honestly
        return None
    rival = next(n for n in TOOL_NAMES if n != picked)

    initial = {n: n[0].upper() for n in TOOL_NAMES}   # e.g. C / W
    strip, settle, rival_peak, rival_layer = [], 0, 0.0, None
    for layer, top5 in enumerate(lens[lens_idx]):
        top1 = classify(top5[0]["t"])
        strip.append(initial.get(top1, "."))
        if top1 != picked:
            settle = layer + 1   # winner must hold from here on
        for e in top5:
            if classify(e["t"]) == rival and e["p"] > rival_peak:
                rival_peak, rival_layer = e["p"], layer

    # decision margin: both tools' probability at the LAST layer's readout —
    # this is (up to readout precision) the distribution the token was
    # sampled from. A near-zero margin = the call was a coin flip, however
    # confident the JSON looks.
    final: dict[str, float] = {}
    for e in lens[lens_idx][-1]:
        tool = classify(e["t"])
        if tool:
            final[tool] = max(final.get(tool, 0.0), e["p"])
    margin = final.get(picked, 0.0) - final.get(rival, 0.0)

    # did the rival surface anywhere EARLIER in the generation?
    early_peak, early_steps = 0.0, 0
    for readout in lens[:lens_idx]:
        p = max((e["p"] for top5 in (readout or []) for e in top5
                 if classify(e["t"]) == rival), default=0.0)
        early_steps += p > 0
        early_peak = max(early_peak, p)

    return {"picked": picked, "rival": rival, "strip": "".join(strip),
            "settle": settle, "n_layers": len(strip), "margin": margin,
            "rival_peak": rival_peak, "rival_layer": rival_layer,
            "early_peak": early_peak, "early_steps": early_steps,
            "trace_id": trace["id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--tag", default="in-two-minds",
                    help="metadata.demo value the traces were tagged with")
    ap.add_argument("--tools", default=",".join(TOOL_NAMES),
                    help="comma-separated pair of tool names to compare")
    args = ap.parse_args()
    TOOL_NAMES[:] = args.tools.split(",")

    index = sorted(get(f"{args.url}/traces")["traces"],
                   key=lambda e: e["ts"], reverse=True)
    seen: dict[str, dict] = {}
    for entry in index:
        tags = entry.get("tags") or {}
        if tags.get("demo") != args.tag or tags.get("case") in seen:
            continue
        seen[tags["case"]] = get(f"{args.url}/traces/{entry['id']}")

    if not seen:
        raise SystemExit("no tagged traces found — run agent.py first "
                         "(and start brainscope with --traces DIR --lens on)")

    print(f"{'case':<15} {'picked':<11} layers (bottom->top)")
    for case, trace in sorted(seen.items()):
        r = analyze(trace)
        if r is None:
            print(f"{case:<15} -- no tool call / no lens data "
                  f"(trace {trace['id']})")
            continue
        settled = (f"settles L{r['settle']}/{r['n_layers']}"
                   if r["settle"] < r["n_layers"] else
                   f"never settles ({r['n_layers']} layers)")
        print(f"{case:<15} {r['picked']:<11} {r['strip']}")
        print(f"{'':<15} {settled}   decision margin {r['margin']:+.2f}   "
              f"rival {r['rival']} peaks p={r['rival_peak']:.2f}"
              + (f" @L{r['rival_layer']}" if r['rival_layer'] is not None else "")
              + (f"   rival alive in {r['early_steps']} earlier steps "
                 f"(p<={r['early_peak']:.2f})" if r["early_steps"] else "")
              + f"   [{r['trace_id']}]")
        j = analyze(trace, "jlens") if trace.get("jlens") else None
        if j:   # server ran with --jlens: the transported "will say later" readout
            print(f"{'':<15} {'J-lens':<11} {j['strip']}   "
                  f"rival {j['rival']} peaks p={j['rival_peak']:.2f}"
                  + (f" @L{j['rival_layer']}" if j['rival_layer'] is not None else ""))

    print("\nReplay any trace in the browser: open brainscope, traces tab, "
          "click the tool-name token, look down the logit-lens column.")


if __name__ == "__main__":
    main()
