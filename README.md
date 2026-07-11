# in-two-minds

**Catch your agent hesitating between two tools — in its activations, not its words.**

A minimal, zero-dependency demo for [brainscope](https://github.com/moudrkat/brainscope):
an agent with two tools (`calculator`, `web_search`) gets five questions —
two clear-cut, three deliberately torn (they need a fact *and* a computation).
The agent answers with a tool call either way; the transcript looks equally
confident every time. The residual stream doesn't.

At the token where the model writes the tool name, the logit lens gives a
per-layer readout of what it would emit if it stopped there. On a clear
question the winning tool is on top almost from the start. On a torn one you
can watch the *other* tool winning through the middle of the stack before the
final layers flip the decision — and `hesitation.py` turns that into two
numbers per request:

- **settle depth** — the first layer from which the winner stays on top.
- **rival peak** — the losing tool's best probability, at the choice token
  and anywhere earlier in the generation.

That pair is the point of the demo: it's a per-request signal you could log
in production next to latency and token counts. A tool call that settled in
the last three layers with the rival at p=0.4 is a different beast than one
that settled at layer 5 — even though both return the same clean JSON.

## Run it

```bash
# 1. the instrument — any HF model brainscope can serve; traces + lens on
pip install brainscope   # or: git clone https://github.com/moudrkat/brainscope && pip install -e .
brainscope --model qwen3-4b --traces traces/ --lens on

# 2. the subject (this repo, stdlib only)
python agent.py
python hesitation.py
```

`--lens on` matters on CPU (the default is auto = on only for CUDA);
without it traces have no per-layer readouts to analyze.

Real output (Qwen3-4B, greedy, `tool_choice: required`):

```
case            picked      layers (bottom->top)
calc_clear      calculator  .............................C.CCCCC
                settles L31/36   decision margin +1.00   rival web_search peaks p=0.00
torn_boiling    web_search  .............................CWWWCWW
                settles L34/36   decision margin +0.91   rival calculator peaks p=0.52 @L33
torn_leap_ms    web_search  ...............................CCCWW
                settles L34/36   decision margin +0.91   rival calculator peaks p=0.85 @L31
```

All three calls come back as equally clean JSON. But `calc_clear` has the
rival at zero everywhere, while `torn_boiling` flip-flops C→W→C into the
last two layers, and in `torn_leap_ms` the calculator is *winning at
p=0.85 five layers before the end* — the decision flipped at the last
moment. How contested these are shows up another way too: changing a
hyphen to an em-dash in the system prompt flipped `torn_leap_ms` to the
other tool. The clear cases don't care.

Then open the brainscope UI, traces tab, click the tool-name token and look
down the logit-lens column — the same story, in color, scrubbing token by
token. This is `torn_leap_ms`, scrubbed to the moment the model wrote `web`:

![replay of torn_leap_ms: layers 24-34 say calculator, the top two flip to web](docs/replay-torn-leap-ms.png)

Layers 24–34 all say *calculator / calculate*; the decision flips in the
last two layers. Side by side, the same column for a clear call and the two
torn ones (bottom = layer 1, top = layer 36 = what gets sampled):

| `calc_clear` → calculator | `torn_boiling` → web_search | `torn_leap_ms` → web_search |
|---|---|---|
| ![calculator wins from layer 24](docs/lens-calc-clear.png) | ![lookup and calculator through the middle, web flip-flops late](docs/lens-torn-boiling.png) | ![calculate all the way up, web only in the last two rows](docs/lens-torn-leap-ms.png) |

(`torn_boiling` bonus: mid-stack the model is reaching for `lookup` — a
tool that doesn't exist.)

## Why this matters for production agents

Tool-call schemas are enforced at decode time these days; the JSON is always
valid. What schemas can't tell you is whether the *decision* behind the JSON
was contested. Watching the layers gives you:

- **a hesitation signal per request** — route contested calls to a validator
  or a human, skip the LLM-judge on confident ones;
- **regression testing for prompt changes** — same battery, compare settle
  depths before/after editing the system prompt or tool descriptions;
- **the silent alternative** — the tool that was never called but kept
  lighting up mid-stack, across your real traffic.

## Honesty note

The numbers come from stored top-5 logit-lens readouts, so they are a lower
bound (a tool missing from a layer's top-5 counts as zero), and the logit
lens itself is a readout convention, not ground truth about the computation.
This demo **illustrates** hesitation on your model and your prompts;
**measuring** it properly means interventions and controls — steer the
decision, rerun, compare. brainscope can do that part too (see its
[steering docs](https://github.com/moudrkat/brainscope/blob/main/docs/steering.md)).
