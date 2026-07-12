"""The killer J-lens demo: the decision is visible before the words.

The agent must open its reply with one neutral sentence ("The user
needs ...", tool names forbidden) and only then call a tool. While that
sentence is being written, the logit lens can only see prose — it reads
"what token comes next". The J-lens (Jacobian lens: what each layer is
pushing the model to say LATER) already reads the upcoming tool choice,
many tokens before the first character of the tool call exists.

Needs a brainscope server running with BOTH readouts:
    brainscope --model ... --traces DIR --lens on --jlens LENS.pt

Usage: python foresight.py [--url http://localhost:8010] [--dump fig/foresight.json]
"""

import argparse
import json
import urllib.parse
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the number.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. 847*391"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web and return the top results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query"}},
            "required": ["query"]}}},
]

# no tool_choice forcing here — the preamble must be free text, so the tool
# decision genuinely lies in the future while the sentence is written.
# The sentence must say WHAT the user wants, never HOW to get it: a verb
# like "calculate" or "search" in the prose would light the logit lens up
# for the trivial reason that it's the literal next word.
SYSTEM = ("You are a helpful assistant with two tools: calculator and "
          "web_search. Your reply MUST have two parts, in this order. "
          "Part 1: exactly one sentence starting with 'The user wants to "
          "know' stating what fact or value they asked for - do not mention "
          "how you would obtain it, no tool names, no verbs like calculate, "
          "compute or search. Part 2: exactly one tool call. Never start "
          "with the tool call.")

CASES = [
    ("fore_calc",   "What is 847 * 391?"),
    ("fore_search", "What was yesterday's closing price of the NVIDIA stock?"),
    ("fore_torn",   "How many milliseconds are in a leap year?"),
    # the minimal pair: both need one physical constant and one division —
    # the model trusts its own physics on one and searches for the other
    ("fore_moon",     "I weigh 70 kg. How much would I weigh on the Moon?"),
    ("fore_sunlight", "How many minutes does sunlight take to reach Earth?"),
]

# the concept each tool lives as, mid-sentence: word family whose summed
# exact probability we track (emergence endpoint sums first tokens of every
# member, with and without leading space)
FAMILIES = {
    "calculator": "calculator,calculate,calculation,calculations,calc,Calculate,Calculator",
    "web_search": "web_search,web,search,Search,searching,browse",
}


def post(url, path, body):
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def analyze(url: str, trace: dict) -> dict | None:
    """Exact per-step p(tool) under both lenses, from the emergence endpoint.

    Needs hidden states stored with the trace ({"hidden": true}, main() turns
    it on for the battery): the stored top-5 readouts are useless here — mid-
    sentence the tool name sits far below the top-5 prose candidates, so the
    lower bound reads 0.00 exactly where the interesting signal lives."""
    toks, off = trace["all_tokens"], trace.get("capture_offset", 0)
    text = "".join(toks)
    call_at = next((i for i, x in enumerate(toks) if "<tool_call>" in x), None)
    if call_at is None or call_at == 0:
        return None      # the model skipped the preamble — nothing to see
    picked = "calculator" if '"calculator"' in text else \
             "web_search" if '"web_search"' in text else None
    if picked is None:
        return None
    # mid-sentence the model doesn't hold the literal token "calculator" —
    # it holds the CONCEPT, spread over the word family (calculate,
    # calculation, …). Track the family's exact summed probability; the
    # system prompt bans these words from the prose, so any mass here is
    # about what comes later, not about the next word.
    em = get(f"{url}/traces/{trace['id']}/emergence"
             f"?token={urllib.parse.quote(FAMILIES[picked])}")
    if not em.get("exact"):
        raise SystemExit("no hidden states with this trace — rerun the "
                         "battery (foresight.py without --skip-run) so "
                         "hidden capture is on")
    lens, jlens = em["series"]["logit_lens"], em["series"].get("jlens")
    # the OTHER tool's family too: on overridden decisions the interesting
    # signal is the rival concept being held and then losing
    rival = "web_search" if picked == "calculator" else "calculator"
    em2 = get(f"{url}/traces/{trace['id']}/emergence"
              f"?token={urllib.parse.quote(FAMILIES[rival])}")
    rlens = em2["series"]["logit_lens"]
    rjlens = em2["series"].get("jlens") or rlens
    steps = [{"i": li + off, "tok": toks[li + off],
              "lens": lens[li], "jlens": (jlens or lens)[li],
              "rlens": rlens[li], "rjlens": rjlens[li]}
             for li in range(len(lens)) if li + off < call_at]
    return {"picked": picked, "rival": rival, "call_at": call_at,
            "preamble": "".join(toks[:call_at]).strip(), "steps": steps,
            "trace_id": trace["id"]}


def bar(p: float, width: int = 10) -> str:
    return ("#" * round(p * width)).ljust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--dump", help="write the analysis as JSON (for fig/render_foresight.py)")
    ap.add_argument("--skip-run", action="store_true",
                    help="only analyze existing traces, don't send new requests")
    args = ap.parse_args()

    if not args.skip_run:
        prior = get(f"{args.url}/traces")   # carries current save/hidden flags
        post(args.url, "/traces/config", {"hidden": True})
        try:
            for case, prompt in CASES:
                print(f"asking: {prompt}")
                post(args.url, "/v1/chat/completions", {
                    "model": "x",
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": prompt}],
                    "tools": TOOLS, "temperature": 0, "max_tokens": 300,
                    "metadata": {"demo": "foresight", "case": case}})
        finally:
            post(args.url, "/traces/config", {"hidden": bool(prior.get("hidden"))})

    index = sorted(get(f"{args.url}/traces")["traces"],
                   key=lambda e: e["ts"], reverse=True)
    results = {}
    for entry in index:
        tags = entry.get("tags") or {}
        if tags.get("demo") != "foresight" or tags.get("case") in results:
            continue
        if not entry.get("has_jlens"):
            raise SystemExit("traces have no J-lens readout — start brainscope "
                             "with --jlens (and --lens on)")
        results[tags["case"]] = analyze(args.url, get(f"{args.url}/traces/{entry['id']}"))

    for case, _ in CASES:
        r = results.get(case)
        if r is None:
            print(f"\n=== {case}: model skipped the preamble — excluded "
                  f"(it happens; rerun or reword)")
            continue
        print(f"\n=== {case}  ->  {r['picked']}  "
              f"(tool call starts at token {r['call_at']})   [{r['trace_id']}]")
        print(f"    “{r['preamble']}”")
        print(f"    {'token':<16} {'logit lens':<22} J-lens")
        for s in r["steps"]:
            print(f"    {s['tok']!r:<16} {bar(s['lens'])} {s['lens']:.2f}   "
                  f"{bar(s['jlens'])} {s['jlens']:.2f}")
        lead_j = next((s["i"] for s in r["steps"] if s["jlens"] >= 0.25), None)
        if lead_j is not None:
            print(f"    J-lens sees {r['picked']} from token {lead_j} — "
                  f"{r['call_at'] - lead_j} tokens before the call is written.")

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"\nwrote {args.dump}")

    print("\nHonesty note: exact readouts (hidden states stored), best layer "
          "per step, p summed over the tool's word family. The prose may not "
          "contain tool-ish words (check the preambles above) - any mass is "
          "a held concept, not the literal next word. The signal peaks at "
          "the verb slot, where the banned word would go; the web_search "
          "case staying dark is the control. Illustrates, not measures: the "
          "intervention (steer & rerun) is the next step.")


if __name__ == "__main__":
    main()
