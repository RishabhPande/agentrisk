"""Score real production-agent sessions (Agent Checkout Lab) with the benchmark's models.

Each real behavioral vector is attached to many draws of legitimate-agent transaction
and network context (local agent: residential IP), so the decline rate reflects how
the models treat this behavior for an otherwise ordinary legitimate customer.
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
import run, sim
from sim import B_FEATS, D_FEATS, T_FEATS, A_FEATS

ROOT = pathlib.Path(__file__).parent
real = pd.read_csv(ROOT / "live" / "real_agent_sessions.csv").sort_values("created").reset_index(drop=True)
real["mode"] = np.where(real.keydown > 0, "keyboard", "click")

N_CTX = 2000
CLOUD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
CTX = sys.argv[2] if len(sys.argv) > 2 else "AU"
rows = []
for seed in range(3):
    tr1 = sim.generate(run.N_TRAIN, seed=seed)
    tra = sim.generate(run.N_AGENT_TRAIN, seed=seed + 100)
    cal = sim.generate(run.N_CAL, seed=seed + 200)
    tr2 = pd.concat([tr1, tra], ignore_index=True)
    BDT = B_FEATS + D_FEATS + T_FEATS
    models = {"Legacy": (BDT, tr1), "Retrained": (BDT, tr2), "AgentGate": (BDT + A_FEATS, tr2)}
    fitted = {}
    for name, (feats, tr) in models.items():
        m = run.fit(tr, feats, seed)
        thr, thr_step = run.threshold_for(m, feats, cal)
        fitted[name] = (m, feats, thr, thr_step)
    # contexts: legitimate local (residential) agents, unverified
    ctx = sim.generate({CTX: N_CTX}, seed=seed + 900, params={"p_cloud_agent": CLOUD})
    hum = sim.generate({"H": N_CTX}, seed=seed + 950)
    for i, r in real.iterrows():
        x = ctx.copy()
        x["is_mobile"] = 0; x["touch_events"] = 0
        x["mousemove"] = r.mousemove
        x["keydown_per_input"] = r.keydown_per_input
        x["iki_mean_ms"] = r.iki_mean_ms if not pd.isna(r.iki_mean_ms) else np.nan
        x["iki_cv"] = (r.iki_sd_ms / r.iki_mean_ms) if (not pd.isna(r.iki_mean_ms) and r.iki_mean_ms) else np.nan
        x["mouse_straightness"] = np.nan if pd.isna(r.mouse_straightness) else r.mouse_straightness
        x["time_to_submit_s"] = r.time_to_submit_ms / 1000.0
        x["untrusted_ratio"] = r.untrusted / r.total
        x["paste_card"] = 0
        x["webdriver"] = int(bool(r.webdriver)); x["headless_ua"] = int(bool(r.ua_headless))
        for name, (m, feats, thr, thr_step) in fitted.items():
            s = m.predict_proba(x[feats])[:, 1]
            if name == "AgentGate":
                pre = run.apply_gate(x, s, thr, thr_step)
                rng = np.random.default_rng(seed * 100 + i)
                final = run.resolve_stepup(x.assign(y=0, cls=CTX), pre, rng)
                dec = float((final == "decline").mean()); step = float((pre == "stepup").mean())
            else:
                dec = float((s >= thr).mean()); step = 0.0
            rows.append(dict(seed=seed, session=r.id, mode=r["mode"], policy=name, decline=dec, stepup=step))
    # reference: same contexts with simulated human behavior swapped in
    for name, (m, feats, thr, thr_step) in fitted.items():
        x = ctx.copy()
        for c in B_FEATS + ["webdriver", "headless_ua"]:
            x[c] = hum[c].values
        s = m.predict_proba(x[feats])[:, 1]
        rows.append(dict(seed=seed, session="sim-human-behavior", mode="human (simulated)", policy=name,
                         decline=float((s >= thr).mean()), stepup=0.0))

out = pd.DataFrame(rows)
out["cloud"] = CLOUD
out["ctx"] = CTX
out.to_csv(ROOT / "results" / f"real_agents_{CTX}_cloud{int(CLOUD)}.csv", index=False)
summ = out.groupby(["mode", "policy"]).agg(decline=("decline", "mean"), stepup=("stepup", "mean")).unstack("policy")
print((summ * 100).round(1).to_string())
print(real[["id", "mode", "mousemove", "click", "keydown", "input", "time_to_submit_ms", "webdriver", "ua_headless"]].to_string())
