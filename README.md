# AgentRisk

Benchmark, telemetry harness and evaluation code for the paper

**Automation Is Not Intent: Measuring and Fixing Fraud-Model Failures on Legitimate AI Shopping Agents**
Rishabh Pande, 2026. [PDF](paper/automation-is-not-intent.pdf) · Preprint DOI: [10.5281/zenodo.23477625](https://doi.org/10.5281/zenodo.23477625) · arXiv: *coming soon*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23477625.svg)](https://doi.org/10.5281/zenodo.23477625)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RishabhPande/agentrisk)

Fraud models learned that "bot-like" means "risky." AI shopping agents now make legitimate purchases with exactly those signals. AgentRisk measures how often fraud models decline legitimate agents, shows how the obvious fixes fail (safe harbor, laundering, hijacking), and evaluates **AgentGate**, a verify-scope-score policy built on agent identity, delegation and mandate scope.

## Quickstart

Try the simulator in seconds (needs only numpy and pandas; run from the repo root):

```bash
pip install numpy pandas
python -c "import sim; df = sim.generate({'H': 200, 'AU': 200, 'AV': 200}, seed=0); print(df.groupby('cls').size().to_dict())"
```

This generates 600 synthetic checkout sessions (200 each of legitimate humans, unverified agents, and verified agents) and prints the per-class counts. The full experiments take longer; see Reproduce below.

Results and demo:

- Hugging Face dataset (results only): https://huggingface.co/datasets/RishabhPande/agentrisk
- Demo Space: https://huggingface.co/spaces/RishabhPande/agentrisk-demo

## What's here

| Path | Contents |
|---|---|
| `sim.py` | AgentRisk-Sim: semi-synthetic checkout-session generator (9 session classes, 36 features). All assumptions are in `PARAMS`. |
| `run.py` | Experiments: `main` (5 seeds, 11 policies), `fewshot`, `sweep` |
| `analyze.py` | Builds every figure, table and in-text number from `results/` |
| `telemetry/` | Instrumented checkout page and Playwright harness that measures what browser-automation primitives emit, plus the raw captures |
| `live/` | The hosted test checkout used with a real production agent, and the 6 recorded sessions (event counts and timings only) |
| `score_real.py`, `analyze_real.py` | Score the real-agent sessions with the benchmark's models |
| `results/` | Result files behind every number in the paper |
| `paper/` | The paper (PDF) |

## Session classes

| Class | Description |
|---|---|
| H | Legitimate human (desktop or mobile, some with autofill) |
| AU | Legitimate unverified agent |
| AV | Legitimate verified agent (signature + signed mandate) |
| FH | Manual fraud |
| FB | Scripted bot fraud (card testing, account takeover) |
| FAU | Fraud via an unverified agent |
| FAV | Laundering through a permissive agent platform |
| FHJ | Hijacked agent (prompt injection) |
| FSP | Spoofed agent signature |

FAU, FAV, FHJ and FSP never appear in training unless a few-shot experiment adds them.

## Reproduce

Python 3.10+.

```bash
pip install -r requirements.txt
python run.py main        # ~8 min on 2 cores
python run.py fewshot     # ~10 min
python run.py sweep       # ~30 min
python analyze.py         # figures, tables, numbers -> paper/gen, paper/figs

# optional: re-measure automation primitives (needs Chromium for Playwright)
python telemetry/capture.py 10

# real-agent scoring (contexts: AU/AV, local/cloud)
for c in 0 1; do for k in AU AV; do python score_real.py $c $k; done; done
python analyze_real.py
```

## Evaluation protocol for new policies

1. Train on the era-1 world (`H`, `FH`, `FB`), optionally plus labeled legitimate agents (`AU`, `AV`).
2. Set the decline threshold on held-out humans so that 2% of legitimate human sessions are declined.
3. Report the share declined for each of the 9 classes, the step-up rate on legitimate classes, and the agent false-decline rate.

## Limitations

Human, fraud, transaction and delegation features are simulated, and every assumption is a parameter. Agent behavior comes from measured automation primitives and from 6 sessions of one production agent. Treat absolute rates as properties of the benchmark, not of any production system. See the paper's Limitations section.

## Citation

```bibtex
@misc{pande2026automation,
  title  = {Automation Is Not Intent: Measuring and Fixing Fraud-Model Failures on Legitimate AI Shopping Agents},
  author = {Pande, Rishabh},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.23477625},
  url    = {https://doi.org/10.5281/zenodo.23477625}
}
```

## License

Code: MIT (see `LICENSE`). Data and results files: CC BY 4.0.
