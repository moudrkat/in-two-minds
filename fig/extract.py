"""Pull the per-layer readouts for the figure from a brainscope trace store.

Takes the newest trace per case (tagged by agent.py), keeps the top-2
entries of both lenses at the tool-choice token, writes figdata.json
next to this script. Then: python render.py [cz]
"""
import argparse
import json
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).parent


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8010")
ap.add_argument("--tag", default="in-two-minds")
args = ap.parse_args()

index = sorted(get(f"{args.url}/traces")["traces"], key=lambda e: e["ts"], reverse=True)
out = {}
for entry in index:
    tags = entry.get("tags") or {}
    if tags.get("demo") != args.tag or tags.get("case") in out:
        continue
    t = get(f"{args.url}/traces/{entry['id']}")
    if not t.get("jlens"):
        continue        # figure needs both lenses — start the server with --jlens
    toks = t["all_tokens"]
    k = next((i for i, x in enumerate(toks)
              if x.strip().lstrip('"') and
              ("calculator".startswith(x.strip().lstrip('"')) or
               x.strip().lstrip('"') == "web")), None)
    if k is None:
        continue
    li = k - t.get("capture_offset", 0)
    out[tags["case"]] = {
        "lens":  [[{"t": e["t"], "p": e["p"]} for e in layer[:2]] for layer in t["lens"][li]],
        "jlens": [[{"t": e["t"], "p": e["p"]} for e in layer[:2]] for layer in t["jlens"][li]]}

(HERE / "figdata.json").write_text(json.dumps(out))
print(f"figdata.json: {sorted(out)}")
