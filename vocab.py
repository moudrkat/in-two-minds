"""Vocabulary census battery: 120 questions, one tool call each.

What words does the model think in, mid-network, before it settles on the
schema name of a tool? This battery generates a large, balanced set of
tool-choice prompts (30 clear-calculator, 30 clear-search, 30 torn
constant+math, 30 torn fact+math), runs them against a brainscope server
with hidden-state capture on, and tags every trace demo=vocab-census.
Analysis lives in fig/vocab_extract.py (exact tuned-lens readouts per
layer at the tool-name token) and fig/render_vocab.py.

Usage: python vocab.py [--url http://localhost:8010] [--limit N]
Runtime guard: stops enrolling new cases after ~55 minutes.
"""

import argparse
import json
import time
import urllib.request

from agent import TOOLS, SYSTEM

CLEAR_CALC = (
    [f"What is {a} * {b}?" for a, b in
     [(847, 391), (263, 749), (912, 384), (577, 216), (438, 655)]] +
    [f"What is {a} + {b}?" for a, b in
     [(48214, 39377), (91724, 8266), (55555, 4445)]] +
    [f"What is {a} minus {b}?" for a, b in [(90210, 4785), (31415, 9265)]] +
    [f"What is {a} divided by {b}?" for a, b in [(986524, 76), (44287, 61)]] +
    [f"What is {p}% of {n}?" for p, n in
     [(15, 2400), (37, 8200), (4.5, 96000), (62, 1250), (8, 71500)]] +
    [f"What is the square root of {n}?" for n in [7744, 55696, 289444]] +
    [f"What is 2 to the power of {k}?" for k in [17, 23, 29]] +
    [f"What is the average of {', '.join(map(str, xs))}?" for xs in
     [(17, 44, 89), (120, 340, 560, 780), (3.5, 7.25, 9.75)]] +
    ["What is 17 factorial?",
     "What is 123456 modulo 789?",
     "Convert the fraction 7/16 to a decimal.",
     "What is (450 + 230) * 12 - 999?"]
)

CLEAR_SEARCH = [
    "What was yesterday's closing price of the NVIDIA stock?",
    "What is the tallest building in the world right now?",
    "Who is the current CEO of OpenAI?",
    "What is the weather forecast for Prague tomorrow?",
    "Who won the Champions League final this year?",
    "What is the latest iPhone model?",
    "Who is the current president of France?",
    "What movies are in theaters this weekend?",
    "What is the current price of Bitcoin?",
    "Who won the Nobel Prize in Physics last year?",
    "What time does the Louvre close today?",
    "What is the newest Tesla model?",
    "Who is the richest person in the world right now?",
    "What were the results of the last Formula 1 race?",
    "Is the Golden Gate Bridge open to traffic today?",
    "What is the current exchange rate of the euro to the dollar?",
    "Who is the head coach of Real Madrid now?",
    "What is trending on social media today?",
    "When is the next SpaceX launch?",
    "What is the current inflation rate in Germany?",
    "Who won the most recent US Open in tennis?",
    "What are today's headlines?",
    "Which country currently holds the EU presidency?",
    "What is the release date of the next Zelda game?",
    "How long is the wait at the Eiffel Tower right now?",
    "What is the air quality in Beijing today?",
    "Who is the current UN Secretary-General?",
    "What is the top song on Spotify this week?",
    "Are there any train strikes in Italy this week?",
    "What is the score of the ongoing cricket match?",
]

TORN_CONST = [
    "I weigh 70 kg. How much would I weigh on the Moon?",
    "I weigh 85 kg. How much would I weigh on Mars?",
    "I weigh 60 kg. How much would I weigh on Jupiter?",
    "How many minutes does sunlight take to reach Earth?",
    "How long does light take to travel from the Sun to Neptune?",
    "How many seconds does sound take to travel 10 kilometers?",
    "What's the average of the boiling points of water and ethanol, in degrees Celsius?",
    "What is the boiling point of water in Fahrenheit minus its freezing point in Fahrenheit?",
    "A dog is 7 years old — how old is that in human years?",
    "A cat is 12 years old — how old is that in human years?",
    "How many Earth days are in 3 Martian years?",
    "How many kilometers is 26.2 miles?",
    "How many pounds is 70 kilograms?",
    "How many liters is 12 US gallons?",
    "What is 100 degrees Fahrenheit in Celsius?",
    "How many centimeters is 6 feet 2 inches?",
    "How far does light travel in one millisecond, in kilometers?",
    "How many times does Earth's circumference fit into the distance to the Moon?",
    "How many Earths would fit across the diameter of the Sun?",
    "How heavy is 5 liters of mercury, in kilograms?",
    "How many grams of gold are in a 24-karat coin weighing 1 troy ounce?",
    "How many breaths does a person take in a year, roughly?",
    "How many heartbeats does a human have in an average lifetime?",
    "How many seconds are in a leap year?",
    "How many milliseconds are in a leap year?",
    "If I drive at 130 km/h, how long does 450 km take?",
    "How many days would it take to walk 500 km at a typical walking pace?",
    "What is the kinetic energy of a 1000 kg car at 90 km/h?",
    "How much does 1 cubic meter of water weigh in tonnes?",
    "How many teaspoons are in half a liter?",
]

TORN_FACT = [
    "How old is the Eiffel Tower in days?",
    "How old is the Great Wall of China in centuries?",
    "How many days ago did the Berlin Wall fall?",
    "How many years passed between the two world wars?",
    "How many days are left until Christmas?",
    "How many days has the current year had so far?",
    "What was the average speed, in km/h, of the marathon world record run?",
    "What is the average speed of the men's 100 m world record, in km/h?",
    "What is the GDP per capita of France?",
    "What is the population density of Japan?",
    "How many people per square kilometer live in Monaco?",
    "How much taller is Mount Everest than K2?",
    "What is the combined length of the Nile and the Amazon rivers?",
    "How many times longer is the Great Wall of China than the English Channel is wide?",
    "How much older is Oxford University than Harvard?",
    "How many Olympic Games have been held since the Eiffel Tower was built?",
    "What fraction of the Earth's population lives in India?",
    "How many years did the Roman Empire last?",
    "How old was Mozart when Beethoven was born?",
    "How many days did the Apollo 11 mission last?",
    "How much deeper is the Mariana Trench than Everest is tall?",
    "How many full moons happen in a decade?",
    "What is the average age of the current US Supreme Court justices?",
    "How many World Cups have been played since the first one?",
    "How old is the Universe in seconds?",
    "How many generations of humans fit in 10,000 years?",
    "What percentage of the EU population lives in Germany?",
    "How much longer is a year on Neptune than on Earth?",
    "How many words does the average person speak in a lifetime?",
    "How old is the oldest living tree, in days?",
]

GROUPS = [("clear_calc", CLEAR_CALC), ("clear_search", CLEAR_SEARCH),
          ("torn_const", TORN_CONST), ("torn_fact", TORN_FACT)]
TIME_BUDGET_S = 55 * 60


def post(url, path, body):
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--limit", type=int, help="run only the first N cases")
    args = ap.parse_args()

    cases = [(f"{g}_{i:02d}", prompt)
             for g, prompts in GROUPS for i, prompt in enumerate(prompts)]
    if args.limit:
        cases = cases[:args.limit]

    prior = get(f"{args.url}/traces")
    post(args.url, "/traces/config", {"hidden": True})
    t0, done = time.time(), 0
    try:
        for case, prompt in cases:
            if time.time() - t0 > TIME_BUDGET_S:
                print(f"time budget hit after {done} cases — stopping")
                break
            msg = post(args.url, "/v1/chat/completions", {
                "model": "x",
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
                "tools": TOOLS, "tool_choice": "required",
                "temperature": 0, "max_tokens": 400,
                "metadata": {"demo": "vocab-census", "case": case},
            })["choices"][0]["message"]
            call = (msg.get("tool_calls") or [{}])[0].get("function", {})
            done += 1
            print(f"[{done}/{len(cases)} {time.time() - t0:5.0f}s] "
                  f"{case}: {call.get('name', '?')}")
    finally:
        post(args.url, "/traces/config", {"hidden": bool(prior.get("hidden"))})
    print(f"\n{done} traces captured. Now: fig/vocab_extract.py (needs the "
          f"hidden states + the tuned lens artifact)")


if __name__ == "__main__":
    main()
