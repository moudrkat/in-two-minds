"""Exact per-layer readouts at the tool-name token for the vocab-census
battery (vocab.py). For every trace tagged demo=vocab-census: top-8 tokens
per layer under the tuned lens and the raw logit lens, plus EXACT summed
probabilities of concept families (schema names vs generic capability
words) — no top-k truncation, full softmax over the vocabulary.

Runs where the model's weights and the traces live (the brainscope host),
e.g. in the experiments container:
  python3 fig/vocab_extract.py
Paths below assume /workspace = trace dir parent, /lens = tuned lens dir.
Writes /workspace/vocab_census.json; render with fig/render_vocab.py.

Same index mapping as tlens_extract.py: stored stack row j (output of
layer j+1, pre-norm) uses translator j+1; row 35 arrives final-normed.
"""
import json
import pathlib

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

TRACES = pathlib.Path("/workspace/traces-demo6")
LENS = pathlib.Path("/lens/qwen3-4b-instruct-2507-tuned-lens/params.pt")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT = pathlib.Path("/workspace/vocab_census.json")

FAMILIES = {
    "schema_calc": ["calculator", "Calculator"],
    "generic_calc": ["calculate", "Calculate", "calc", "calculation",
                     "compute", "Compute", "computing", "math", "evaluate"],
    "schema_web": ["web_search", "web", "Web"],
    "generic_web": ["search", "Search", "lookup", "look", "query", "Query",
                    "google", "Google", "api", "API", "database", "find",
                    "fetch", "browse", "retrieve", "internet"],
}

cfg = AutoConfig.from_pretrained(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
sd = torch.load(LENS, map_location="cpu")
translators = {int(k.split(".")[0]): (sd[f"{k.split('.')[0]}.weight"].float(),
                                      sd[f"{k.split('.')[0]}.bias"].float())
               for k in sd if k.endswith(".weight")}

from huggingface_hub import snapshot_download
mdir = pathlib.Path(snapshot_download(MODEL, allow_patterns=["*.json"]))
index = json.loads((mdir / "model.safetensors.index.json").read_text())["weight_map"]
head_key = "lm_head.weight" if "lm_head.weight" in index else "model.embed_tokens.weight"
weights = {}
for name in ("model.norm.weight", head_key):
    shard = pathlib.Path(snapshot_download(MODEL, allow_patterns=[index[name]])) / index[name]
    with safe_open(str(shard), framework="pt") as f:
        weights[name] = f.get_tensor(name).float()
norm_w, head_w = weights["model.norm.weight"], weights[head_key]

# family word -> set of first-token ids, with and without leading space
fam_ids = {}
for fam, words in FAMILIES.items():
    ids = set()
    for w in words:
        for v in (w, " " + w):
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    fam_ids[fam] = sorted(ids)
print({f: len(i) for f, i in fam_ids.items()})


def rmsnorm(h):
    return h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w


def probs(z, normed_already=False):
    zn = z if normed_already else rmsnorm(z)
    return torch.softmax(zn @ head_w.T, dim=-1)


def tool_token_index(all_tokens):
    for i, x in enumerate(all_tokens):
        s = x.strip().lstrip('"')
        if s and ("calculator".startswith(s) or s == "web"):
            return i
    return None


out = {}
for f in sorted(TRACES.glob("*.json")):
    t = json.loads(f.read_text())
    tags = t.get("tags") or {}
    if tags.get("demo") != "vocab-census" or not t.get("has_hidden"):
        continue
    case = tags.get("case")
    hf = TRACES / f"{t['id']}.hidden.pt"
    if case in out or not hf.exists():
        continue
    k = tool_token_index(t["all_tokens"])
    if k is None:
        print(f"{case}: no tool token, skipped")
        continue
    hidden = torch.load(hf, map_location="cpu").float()
    li = k - t.get("capture_offset", 0)
    if not (0 <= li < hidden.shape[0]):
        print(f"{case}: tool token outside captured steps, skipped")
        continue
    h = hidden[li]
    picked = "calculator" if "calculator".startswith(
        t["all_tokens"][k].strip().lstrip('"')) else "web_search"

    rec = {"group": case.rsplit("_", 1)[0], "picked": picked,
           "top8": {"tuned": [], "raw": []},
           "fam": {"tuned": {fam: [] for fam in FAMILIES},
                   "raw": {fam: [] for fam in FAMILIES}}}
    for j in range(36):
        variants = {}
        variants["raw"] = probs(h[j], normed_already=(j == 35))
        if j < 35:
            W, b = translators[j + 1]
            variants["tuned"] = probs(h[j] + h[j] @ W.T + b)
        else:
            variants["tuned"] = variants["raw"]
        for key, p in variants.items():
            top = p.topk(8)
            rec["top8"][key].append(
                [{"t": tok.decode(int(i)), "p": round(float(v), 4)}
                 for v, i in zip(top.values, top.indices)])
            for fam, ids in fam_ids.items():
                rec["fam"][key][fam].append(round(float(p[ids].sum()), 4))
    out[case] = rec
    print(f"{case}: {picked}")

OUT.write_text(json.dumps(out))
print(f"wrote {OUT}: {len(out)} cases")
