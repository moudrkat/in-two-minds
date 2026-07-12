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


def big_groups(n_per_group=250, seed=42):
    """Template-generated battery, deterministic. ~4×250 prompts on top of
    the hand-written 120 (case names get a b prefix so the sets can merge)."""
    import random
    rng = random.Random(seed)
    ri = lambda a, b: rng.randint(a, b)

    calc = []
    for _ in range(35): calc.append(f"What is {ri(103, 987)} * {ri(103, 987)}?")
    for _ in range(30): calc.append(f"What is {ri(10_000, 99_000)} + {ri(1_000, 99_000)}?")
    for _ in range(30): calc.append(f"What is {ri(50_000, 99_000)} minus {ri(1_000, 49_000)}?")
    for _ in range(30): calc.append(f"What is {ri(10_000, 999_000)} divided by {ri(7, 97)}?")
    for _ in range(30): calc.append(f"What is {ri(3, 95)}% of {ri(200, 99_000)}?")
    for _ in range(25):
        r = ri(37, 700); calc.append(f"What is the square root of {r * r}?")
    for _ in range(25): calc.append(f"What is {ri(2, 9)} to the power of {ri(7, 30)}?")
    for _ in range(25):
        xs = sorted(rng.sample(range(10, 999), ri(3, 5)))
        calc.append(f"What is the average of {', '.join(map(str, xs))}?")
    for _ in range(20): calc.append(f"What is {ri(10_000, 999_999)} modulo {ri(7, 999)}?")

    COMPANIES = ["Apple", "Microsoft", "Tesla", "Amazon", "Google", "Meta",
                 "Netflix", "Intel", "AMD", "Samsung", "Sony", "Nike",
                 "Coca-Cola", "Boeing", "Airbus", "Shell", "Toyota", "Volkswagen"]
    CITIES = ["Prague", "Vienna", "Tokyo", "Sydney", "Toronto", "Oslo",
              "Lisbon", "Seoul", "Nairobi", "Lima", "Reykjavik", "Athens",
              "Dublin", "Warsaw", "Bangkok", "Cairo"]
    EVENTS = ["SpaceX launch", "Apple keynote", "UEFA Champions League final",
              "Wimbledon final", "Olympic Games", "G7 summit", "solar eclipse",
              "Eurovision final", "Oscars ceremony", "Comic-Con"]
    THINGS = ["iPhone", "Samsung Galaxy", "PlayStation", "Xbox", "Tesla Model",
              "MacBook", "Kindle", "GoPro", "Fitbit", "ThinkPad"]
    search = []
    search += [f"What was yesterday's closing price of the {c} stock?" for c in COMPANIES]
    search += [f"Who is the current CEO of {c}?" for c in COMPANIES]
    search += [f"What is the weather forecast for {c} tomorrow?" for c in CITIES]
    search += [f"What is the air quality in {c} today?" for c in CITIES]
    search += [f"What time is it right now in {c}?" for c in CITIES[:10]]
    search += [f"When is the next {e}?" for e in EVENTS]
    search += [f"Who won the most recent {e.replace('the next ', '')}?" for e in
               ["Formula 1 race", "Super Bowl", "World Series", "Tour de France",
                "Boston Marathon", "Ballon d'Or", "Nobel Peace Prize", "Grammy for album of the year"]]
    search += [f"What is the latest {t} model?" for t in THINGS]
    search += [f"What is the current price of {x}?" for x in
               ["Bitcoin", "Ethereum", "gold per ounce", "silver per ounce",
                "crude oil per barrel", "a Big Mac in Switzerland"]]
    search += [f"Who is the current president of {c}?" for c in
               ["France", "Brazil", "Argentina", "South Korea", "Mexico",
                "Poland", "Finland", "Kenya", "Indonesia", "Chile"]]
    search += [f"What is the current exchange rate of the {a} to the {b}?" for a, b in
               [("euro", "dollar"), ("yen", "dollar"), ("pound", "euro"),
                ("Swiss franc", "dollar"), ("Czech koruna", "euro"),
                ("Canadian dollar", "US dollar"), ("yuan", "dollar"),
                ("rupee", "dollar"), ("real", "dollar"), ("peso", "dollar")]]
    search += [f"Who won this year's {a}?" for a in
               ["Pulitzer Prize for fiction", "Booker Prize", "Palme d'Or",
                "Turing Award", "Fields Medal", "Grammy for record of the year",
                "Emmy for best drama", "MotoGP championship",
                "Stanley Cup", "NBA championship"]]
    search += [f"What is the current inflation rate in {c}?" for c in
               ["Germany", "the US", "the UK", "Japan", "Turkey", "Argentina",
                "the eurozone", "the Czech Republic"]]
    search += [f"Are there any {x} this week?" for x in
               ["train strikes in Italy", "train strikes in Germany",
                "airline strikes in France", "road closures in downtown Vienna",
                "protests planned in Paris", "public holidays in Spain"]]
    search += [f"What is the ticket price for {p} right now?" for p in
               ["the Louvre", "Disneyland Paris", "the Empire State Building",
                "Legoland Billund", "the London Eye", "the Vatican Museums"]]
    search += [f"How long is the current wait at {p}?" for p in
               ["the Eiffel Tower", "the Uffizi Gallery", "the DMV in Los Angeles",
                "the Anne Frank House"]]
    search += [f"What is the score of the ongoing {s} match?" for s in
               ["cricket", "Premier League", "NBA", "NHL", "Champions League"]]
    search += [f"Which {x} is number one right now?" for x in
               ["song on Spotify", "movie at the box office", "book on the NYT bestseller list",
                "app in the App Store", "team in the Bundesliga"]]
    search += ["What are today's headlines?", "What is trending on social media today?",
               "Which movies open in theaters this week?",
               "Who is leading the Premier League right now?",
               "Is the Suez Canal open to traffic today?",
               "Is the Golden Gate Bridge open to traffic today?",
               "When does the next train from Prague to Vienna leave?",
               "What is the pollen forecast for tomorrow?",
               "Which country currently holds the EU presidency?",
               "Who is the current UN Secretary-General?",
               "What is the current population of Earth?",
               "How many people are in space right now?"]
    search = search[:n_per_group]

    PLANETS = [("the Moon", None), ("Mars", None), ("Jupiter", None),
               ("Venus", None), ("Mercury", None), ("Neptune", None),
               ("Saturn", None), ("Pluto", None)]
    const = []
    for p, _ in PLANETS:
        for _ in range(4):
            const.append(f"I weigh {ri(50, 110)} kg. How much would I weigh on {p}?")
    const += [f"How long does light take to travel from the Sun to {p}?"
              for p, _ in PLANETS]
    for _ in range(12): const.append(f"How many seconds does sound take to travel {ri(2, 90)} kilometers?")
    for _ in range(12): const.append(f"If I drive at {ri(60, 140)} km/h, how long does {ri(80, 900)} km take?")
    for _ in range(10): const.append(f"How many kilometers is {ri(5, 500)} miles?")
    for _ in range(10): const.append(f"How many pounds is {ri(40, 200)} kilograms?")
    for _ in range(10): const.append(f"How many liters is {ri(3, 40)} US gallons?")
    for _ in range(10): const.append(f"What is {ri(50, 400)} degrees Fahrenheit in Celsius?")
    for _ in range(10): const.append(f"How many centimeters is {ri(4, 7)} feet {ri(0, 11)} inches?")
    for _ in range(8): const.append(f"A dog is {ri(2, 15)} years old — how old is that in human years?")
    for _ in range(8): const.append(f"A cat is {ri(2, 18)} years old — how old is that in human years?")
    for _ in range(8): const.append(f"How heavy is {ri(2, 20)} liters of mercury, in kilograms?")
    for _ in range(8): const.append(f"How many Earth days are in {ri(2, 10)} Martian years?")
    for _ in range(10): const.append(f"How many acres is {ri(2, 90)} hectares?")
    for _ in range(10): const.append(f"How many km/h is {ri(5, 60)} knots?")
    for _ in range(10): const.append(f"How many kilograms is {ri(6, 30)} stone?")
    for _ in range(10): const.append(f"How many milliliters is {ri(2, 12)} cups?")
    for _ in range(10): const.append(f"How many bars is {ri(15, 120)} psi?")
    for _ in range(8): const.append(f"A horse is {ri(2, 30)} years old — how old is that in human years?")
    for _ in range(8): const.append(f"How many calories are in {ri(50, 400)} grams of pure sugar?")
    for _ in range(8): const.append(f"How many minutes does the ISS need for {ri(2, 10)} orbits of Earth?")
    for _ in range(8): const.append(f"How many seconds would a fall from {ri(100, 2000)} meters take, ignoring air resistance?")
    const += ["How many seconds are in a leap year?",
              "How many milliseconds are in a leap year?",
              "What's the average of the boiling points of water and ethanol, in degrees Celsius?",
              "How far does light travel in one millisecond, in kilometers?",
              "How many breaths does a person take in a year, roughly?",
              "How many heartbeats does a human have in an average lifetime?",
              "How much does 1 cubic meter of water weigh in tonnes?",
              "How many teaspoons are in half a liter?"]
    const = const[:n_per_group]

    LANDMARKS = ["the Eiffel Tower", "the Brooklyn Bridge", "the Sydney Opera House",
                 "Big Ben", "the Statue of Liberty", "the Golden Gate Bridge",
                 "the Empire State Building", "Notre-Dame cathedral", "the Panama Canal",
                 "the Trans-Siberian Railway", "Machu Picchu's rediscovery",
                 "the Berlin TV tower", "the Hoover Dam", "the Suez Canal",
                 "the Great Pyramid of Giza", "Stonehenge", "the Colosseum",
                 "Hagia Sophia", "the Taj Mahal", "the Forbidden City"]
    RECORDS = ["marathon", "men's 100 m sprint", "women's 100 m sprint",
               "one-mile run", "10,000 m run", "Tour de France",
               "English Channel swim", "Everest ascent"]
    COUNTRIES = ["Japan", "Monaco", "France", "Brazil", "Canada", "India",
                 "Australia", "Iceland", "Nigeria", "Bangladesh", "Singapore",
                 "Mongolia", "the Netherlands", "Egypt", "Norway", "Vietnam"]
    fact = []
    fact += [f"How old is {l} in days?" for l in LANDMARKS]
    fact += [f"How old is {l} in weeks?" for l in LANDMARKS[:10]]
    fact += [f"What was the average speed, in km/h, of the {r} world record?" for r in RECORDS]
    fact += [f"What is the population density of {c}?" for c in COUNTRIES]
    fact += [f"What is the GDP per capita of {c}?" for c in COUNTRIES]
    fact += [f"How many days are left until {h}?" for h in
             ["Christmas", "New Year's Eve", "Halloween", "Easter",
              "the next Summer Olympics", "the next FIFA World Cup"]]
    fact += [f"How many days ago did {e}?" for e in
             ["the Berlin Wall fall", "the Titanic sink", "Apollo 11 land on the Moon",
              "World War II end", "the French Revolution begin",
              "Czechoslovakia split into two countries"]]
    fact += [f"How much taller is {a} than {b}?" for a, b in
             [("Mount Everest", "K2"), ("Mont Blanc", "Mount Olympus"),
              ("the Burj Khalifa", "the Eiffel Tower"), ("Denali", "Mount Whitney"),
              ("Kilimanjaro", "Mount Fuji")]]
    fact += [f"How many times longer is {a} than {b}?" for a, b in
             [("the Nile", "the Thames"), ("the Amazon", "the Seine"),
              ("the Great Wall of China", "Hadrian's Wall"),
              ("the Danube", "the Vltava")]]
    fact += [f"How old is {l} in hours?" for l in LANDMARKS[10:]]
    fact += [f"How many years passed between the invention of {a} and {b}?" for a, b in
             [("the telephone", "the internet"), ("the printing press", "the typewriter"),
              ("the steam engine", "the jet engine"), ("photography", "cinema"),
              ("the telegraph", "the smartphone"), ("the wheel", "the bicycle"),
              ("penicillin", "the mRNA vaccine"), ("the light bulb", "the LED")]]
    fact += [f"How old was {p} when {e}?" for p, e in
             [("Mozart", "Beethoven was born"), ("Einstein", "he published special relativity"),
              ("Newton", "he published the Principia"), ("Marie Curie", "she won her first Nobel Prize"),
              ("Darwin", "he published On the Origin of Species"),
              ("Cleopatra", "Caesar was assassinated"), ("Gandhi", "India gained independence")]]
    fact += [f"What percentage of the world's population lives in {c}?" for c in
             ["China", "India", "the EU", "Africa", "the US", "Indonesia"]]
    fact += [f"How many people per square kilometer live in {c}?" for c in
             ["Macau", "Hong Kong", "Malta", "Gibraltar", "the Vatican"]]
    fact += [f"What is the average age of {g}?" for g in
             ["the current US Supreme Court justices", "current EU heads of state",
              "the current UK cabinet", "Nobel laureates at the time of the award"]]
    fact += ["How many years did the Roman Empire last?",
             "How many years passed between the two world wars?",
             "How much older is Oxford University than Harvard?",
             "How many Olympic Games have been held since the Eiffel Tower was built?",
             "How old is the Universe in seconds?",
             "How many full moons happen in a decade?",
             "How many World Cups have been played since the first one?",
             "How many days did the Apollo 11 mission last?",
             "How much deeper is the Mariana Trench than Everest is tall?",
             "How many generations of humans fit in 10,000 years?"]
    fact = fact[:n_per_group]

    return [("clear_calc", calc[:n_per_group]), ("clear_search", search),
            ("torn_const", const), ("torn_fact", fact)]


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
    ap.add_argument("--big", action="store_true",
                    help="template-generated ~1000-case battery (case prefix b)")
    args = ap.parse_args()

    groups = big_groups() if args.big else GROUPS
    tag = "b" if args.big else ""
    cases = [(f"{g}_{tag}{i:03d}", prompt)
             for g, prompts in groups for i, prompt in enumerate(prompts)]
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
