"""Tuned-lens readouts for the in-two-minds figure, computed offline from
stored hidden states.

Reads the newest trace per case (demo=in-two-minds) from a trace dir,
applies the tuned-lens translators (Belrose et al.; residual convention
h + A_l h + b_l, then final norm + lm_head) at the tool-choice token, and
writes tlens_figdata.json: {case: {"tlens": [[{t,p} top-2] per layer]}}.

Index mapping: brainscope stores hidden_states[1:] (outputs of layers
1..36); translator l of the lens was fitted on pre-norm hidden_states[l],
so stack row j uses translator j+1 for j in 0..34. Row 35 arrives already
final-normed (HF convention) and reads out through the head directly.

Sanity check: recomputes the raw logit lens per row and reports top-1
agreement with the readouts recorded in the trace.
"""
import json
import pathlib
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

TRACES = pathlib.Path("/workspace/traces-demo6")
LENS = pathlib.Path("/lens/qwen3-4b-instruct-2507-tuned-lens/params.pt")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT = pathlib.Path("/workspace/tlens_figdata.json")

cfg = AutoConfig.from_pretrained(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
sd = torch.load(LENS, map_location="cpu")
translators = {int(k.split(".")[0]): None for k in sd}
translators = {l: (sd[f"{l}.weight"].float(), sd[f"{l}.bias"].float())
               for l in translators}
print(f"translators: {len(translators)}, d={cfg.hidden_size}, "
      f"layers={cfg.num_hidden_layers}, eps={cfg.rms_norm_eps}")

# final norm + head weights straight from the checkpoint shards
from huggingface_hub import snapshot_download
mdir = pathlib.Path(snapshot_download(MODEL, allow_patterns=["*.json"]))
index = json.loads((mdir / "model.safetensors.index.json").read_text())["weight_map"]
head_key = "lm_head.weight" if "lm_head.weight" in index else "model.embed_tokens.weight"
want = {"model.norm.weight", head_key}
weights = {}
for name in want:
    shard = pathlib.Path(snapshot_download(MODEL, allow_patterns=[index[name]])) / index[name]
    with safe_open(str(shard), framework="pt") as f:
        weights[name] = f.get_tensor(name).float()
norm_w = weights["model.norm.weight"]
head_w = weights[head_key]
print(f"head: {head_key} {tuple(head_w.shape)}")


def rmsnorm(h):
    return h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w


def readout(z, normed_already=False):
    zn = z if normed_already else rmsnorm(z)
    p = torch.softmax(zn @ head_w.T, dim=-1)
    top = p.topk(2)
    return [{"t": tok.decode(int(i)), "p": round(float(v), 4)}
            for v, i in zip(top.values, top.indices)]


def tool_token_index(all_tokens):
    for i, x in enumerate(all_tokens):
        s = x.strip().lstrip('"')
        if s and ("calculator".startswith(s) or s == "web"):
            return i
    return None


newest = {}
for f in TRACES.glob("*.json"):
    t = json.loads(f.read_text())
    tags = t.get("tags") or {}
    if tags.get("demo") != "in-two-minds" or not t.get("has_hidden"):
        continue
    case = tags.get("case")
    if case and (case not in newest or t.get("ts", 0) > newest[case].get("ts", 0)):
        newest[case] = t

out = {}
for case, t in sorted(newest.items()):
    hf = TRACES / f"{t['id']}.hidden.pt"
    hidden = torch.load(hf, map_location="cpu").float()  # [steps, 36, d]
    k = tool_token_index(t["all_tokens"])
    if k is None:
        print(f"{case}: no tool token found, skipped")
        continue
    li = k - t.get("capture_offset", 0)
    h = hidden[li]  # [36, d]

    # sanity: raw logit lens vs the trace's recorded readout
    agree = 0
    for j in range(36):
        mine = readout(h[j], normed_already=(j == 35))[0]["t"]
        if mine == t["lens"][li][j][0]["t"]:
            agree += 1
    grid = []
    for j in range(36):
        if j < 35:
            W, b = translators[j + 1]
            z = h[j] + h[j] @ W.T + b
            grid.append(readout(z))
        else:
            grid.append(readout(h[j], normed_already=True))
    out[case] = {"tlens": grid}
    tops = [g[0]["t"].strip() for g in grid]
    print(f"{case}: token {t['all_tokens'][k]!r} raw-agreement {agree}/36, "
          f"tuned top-1 tail: {tops[-6:]}")

OUT.write_text(json.dumps(out))
print(f"wrote {OUT} ({sorted(out)})")
