"""A deliberately torn agent: two tools, five questions, zero dependencies.

Points an OpenAI-style chat request at a running brainscope server
(https://github.com/moudrkat/brainscope) and asks questions ranging from
clearly-calculator to clearly-web-search — with a few traps in the middle
where both tools are defensible. Every request is tagged, so the traces
brainscope persists can be found and analyzed afterwards by hesitation.py.

Usage:
    # terminal 1 — the instrument (lens capture ON, traces ON):
    brainscope --model qwen3-4b --traces traces/ --lens on
    # terminal 2 — the subject:
    python agent.py [--url http://localhost:8010]
"""

import argparse
import json
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

SYSTEM = ("You are a helpful assistant with two tools: calculator and "
          "web_search. Answer by calling exactly one tool — pick the one "
          "that gets you furthest.")

# From clear-cut to genuinely torn. The torn ones need a fact AND a
# computation, so whichever tool the model picks, the other was a live option.
# (Battery tuned on Qwen3-4B; the boundary sits elsewhere for other models —
# add your own cases and watch where YOUR model gets torn.)
CASES = [
    ("calc_clear",    "What is 847 * 391?"),
    ("search_clear",  "What was yesterday's closing price of the NVIDIA stock?"),
    ("torn_boiling",  "What's the average of the boiling points of water and "
                      "ethanol, in degrees Celsius?"),
    ("torn_leap_ms",  "How many milliseconds are in a leap year?"),
    ("torn_eiffel",   "How old is the Eiffel Tower in days?"),
    ("calc_clear2",   "What is 15% of 2400?"),
    ("search_clear2", "What is the tallest building in the world right now?"),
    ("torn_dog",      "A dog is 7 years old — how old is that in human years?"),
    ("torn_sunlight", "How many minutes does sunlight take to reach Earth?"),
    ("torn_moon",     "I weigh 70 kg. How much would I weigh on the Moon?"),
    ("torn_marathon", "What was the average speed, in km/h, of the marathon "
                      "world record run?"),
    ("torn_gdp",      "What is the GDP per capita of France?"),
]


def chat(url: str, case: str, prompt: str) -> dict:
    body = {
        "model": "whatever-brainscope-serves",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "tools": TOOLS,
        "tool_choice": "required",  # no essay escape hatch: brainscope seeds
                                    # the call syntax, the model only picks
        "temperature": 0,           # the hesitation we want lives in the
        "max_tokens": 400,          # activations, not in sampling noise
        "raw": True,                # keep <think> in the response too
        "metadata": {"demo": "in-two-minds", "case": case},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8010")
    args = ap.parse_args()

    for case, prompt in CASES:
        print(f"\n=== {case}: {prompt}")
        msg = chat(args.url, case, prompt)["choices"][0]["message"]
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            print(f"    -> {fn['name']}({fn['arguments']})")
        if not msg.get("tool_calls"):
            print(f"    (no tool call) {msg.get('content', '')!r:.200}")

    print("\nDone. Now: python hesitation.py --url", args.url)


if __name__ == "__main__":
    main()