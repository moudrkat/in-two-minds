"""Is the tuned lens actually good? Per-layer KL(final || lens readout),
averaged over every captured position of the in-two-minds traces.

The final row of the stored stack (already final-normed) read through the
head IS the model's output distribution — the target. A useful tuned lens
must have much lower KL than the raw logit lens at every depth; a lazy one
(translators ~ identity) sits on the raw curve. Also loads the step-50
checkpoint: if KL kept dropping 50 -> 250, longer training would help.
"""
import json
import pathlib

import torch
from safetensors import safe_open
from transformers import AutoConfig

TRACES = pathlib.Path("/workspace/traces-demo6")
LDIR = pathlib.Path("/lens/qwen3-4b-instruct-2507-tuned-lens")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

cfg = AutoConfig.from_pretrained(MODEL)


def load_translators(sd):
    if "lens" in sd:                       # trainer snapshot
        sd = sd["lens"]
    sd = {k.removeprefix("layer_translators."): v for k, v in sd.items()}
    return {int(k.split(".")[0]): (sd[f"{k.split('.')[0]}.weight"].float(),
                                   sd[f"{k.split('.')[0]}.bias"].float())
            for k in sd if k.endswith(".weight") and k.split(".")[0].isdigit()}


final_tl = load_translators(torch.load(LDIR / "params.pt", map_location="cpu"))
try:
    snap = torch.load(LDIR / "checkpoints/snapshot_50.pth", map_location="cpu",
                      weights_only=False)
    mid_tl = load_translators(snap)
    print(f"snapshot_50 loaded: {len(mid_tl)} translators")
except Exception as e:
    mid_tl = None
    print(f"snapshot_50 not usable ({type(e).__name__}: {e}) — skipping")

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


def rmsnorm(h):
    return h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w


def logprobs(z, normed_already=False):
    zn = z if normed_already else rmsnorm(z)
    return torch.log_softmax(zn @ head_w.T, dim=-1)


traces = []
for f in TRACES.glob("*.json"):
    t = json.loads(f.read_text())
    if (t.get("tags") or {}).get("demo") == "in-two-minds" and t.get("has_hidden"):
        traces.append(t)

kl_raw = torch.zeros(35)
kl_tuned = torch.zeros(35)
kl_mid = torch.zeros(35)
n = 0
for t in traces:
    hidden = torch.load(TRACES / f"{t['id']}.hidden.pt", map_location="cpu").float()
    for pos in range(hidden.shape[0]):
        h = hidden[pos]                       # [36, d]
        target = logprobs(h[35], normed_already=True)
        p_t = target.exp()
        for j in range(35):
            lp = logprobs(h[j])
            kl_raw[j] += float((p_t * (target - lp)).sum())
            W, b = final_tl[j + 1]
            lp = logprobs(h[j] + h[j] @ W.T + b)
            kl_tuned[j] += float((p_t * (target - lp)).sum())
            if mid_tl is not None:
                W, b = mid_tl[j + 1]
                lp = logprobs(h[j] + h[j] @ W.T + b)
                kl_mid[j] += float((p_t * (target - lp)).sum())
        n += 1

print(f"\npositions: {n} (from {len(traces)} traces)")
print(f"{'layer':>6} {'raw':>7} {'tuned':>7}" + (f" {'@step50':>8}" if mid_tl else ""))
for j in range(35):
    row = f"{'L' + str(j + 1):>6} {kl_raw[j] / n:7.3f} {kl_tuned[j] / n:7.3f}"
    if mid_tl is not None:
        row += f" {kl_mid[j] / n:8.3f}"
    print(row)
print(f"{'mean':>6} {kl_raw.mean() / n:7.3f} {kl_tuned.mean() / n:7.3f}"
      + (f" {kl_mid.mean() / n:8.3f}" if mid_tl else ""))
