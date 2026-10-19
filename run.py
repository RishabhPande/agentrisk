"""Experiments for the AgentRisk benchmark. Writes results/*.csv and results/*.json."""
import json, pathlib, sys, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import sim
from sim import B_FEATS, D_FEATS, T_FEATS, A_ID, A_DEL, A_CON, A_FEATS, LEGIT, ERA2_FRAUD

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).parent
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
tel = ROOT / "telemetry" / "telemetry_summary.csv"
if tel.exists():
    sim.load_telemetry(tel)

SEEDS = [0, 1, 2, 3, 4]
HUMAN_DECLINE = 0.02          # operating point: 2% of legit human sessions declined
STEPUP_PASS = dict(legit=0.92, FHJ=0.10, fraud=0.05)
SCOPE_TOL = 0.10
N_TRAIN = dict(H=150_000, FH=1_500, FB=1_000)
N_AGENT_TRAIN = dict(AU=8_000, AV=8_000)
N_TEST = dict(H=150_000, AU=10_000, AV=10_000, FH=2_000, FB=2_000, FAU=2_000, FAV=2_000, FHJ=2_000, FSP=2_000)
N_CAL = dict(H=100_000)

LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit(df, feats, seed):
    m = lgb.LGBMClassifier(**LGB, random_state=seed)
    m.fit(df[feats], df["y"])
    return m


def automation_flag(df):
    return ((df.webdriver == 1) | (df.untrusted_ratio > 0) | (df.headless_ua == 1) |
            ((df.keydown_per_input < 0.05) & (df.mousemove < 20) & (df.is_mobile == 0))).astype(int)


def decide_ml(score, thr):
    return np.where(score >= thr, "decline", "approve")


def apply_gate(df, score, thr, thr_step):
    """AgentGate: verify -> scope -> score.  Returns decision per session."""
    dec = np.where(score >= thr, "decline", "approve").astype(object)
    unverified_auto = (df.sig_valid == 0) & (automation_flag(df) == 1)
    # Unattested automation in the grey band is stepped up instead of approved
    dec = np.where(unverified_auto & (score >= thr_step) & (dec == "approve"), "stepup", dec)
    verified = df.sig_valid == 1
    unbound = verified & (df.cred_bound_to_principal == 0)
    scope_viol = verified & ((df.merchant_in_scope == 0) | (df.category_in_scope == 0) |
                             (df.amount_over_cap > SCOPE_TOL) | (df.ship_to_principal == 0))
    dec = np.where((unbound | scope_viol) & (dec == "approve"), "stepup", dec)
    dec = np.where((df.sig_present == 1) & (df.sig_valid == 0), "decline", dec)
    dec = np.where((df.mandate_present == 1) & (df.mandate_valid == 0), "decline", dec)
    return dec


def resolve_stepup(df, dec, rng):
    """Step-up: legit principals usually confirm; fraudsters/hijack victims usually don't."""
    p = np.where(df.y == 0, STEPUP_PASS["legit"], np.where(df.cls == "FHJ", STEPUP_PASS["FHJ"], STEPUP_PASS["fraud"]))
    passed = rng.random(len(df)) < p
    return np.where(dec == "stepup", np.where(passed, "approve", "decline"), dec)


def per_class(df, dec_pre, dec_final):
    out = {}
    for c in sim.ALL:
        m = (df.cls == c).values
        if m.sum() == 0:
            continue
        out[c] = dict(decline=float((dec_final[m] == "decline").mean()),
                      stepup=float((dec_pre[m] == "stepup").mean()),
                      hard_decline=float((dec_pre[m] == "decline").mean()))
    return out


def threshold_for(model, feats, cal, rate=HUMAN_DECLINE):
    s = model.predict_proba(cal[feats])[:, 1]
    return float(np.quantile(s, 1 - rate)), float(np.quantile(s, 1 - 5 * rate))


def run_seed(seed, params=None, few_shot=0, only=None):
    rng = np.random.default_rng(1000 + seed)
    tr1 = sim.generate(N_TRAIN, seed=seed, params=params)                       # era-1 world
    tra = sim.generate(N_AGENT_TRAIN, seed=seed + 100, params=params)          # labeled legit agents
    cal = sim.generate(N_CAL, seed=seed + 200, params=params)
    te = sim.generate(N_TEST, seed=seed + 300, params=params)
    extra = None
    if few_shot:
        extra = sim.generate({c: few_shot for c in ERA2_FRAUD}, seed=seed + 400, params=params)
    tr2 = pd.concat([tr1, tra] + ([extra] if extra is not None else []), ignore_index=True)

    BDT = B_FEATS + D_FEATS + T_FEATS
    configs = {
        "Legacy":               dict(feats=BDT, train=tr1 if extra is None else pd.concat([tr1, extra]), mode="ml"),
        "Legacy+Allowlist":     dict(feats=BDT, train=tr1 if extra is None else pd.concat([tr1, extra]), mode="allow"),
        "Retrained":            dict(feats=BDT, train=tr2, mode="ml"),
        "Retrained+Allowlist":  dict(feats=BDT, train=tr2, mode="allow"),
        "Identity-aware":       dict(feats=BDT + A_ID, train=tr2, mode="ml"),
        "AgentGate":            dict(feats=BDT + A_FEATS, train=tr2, mode="gate"),
        "AgentGate-invariant":  dict(feats=["paste_card", "is_mobile"] + T_FEATS + A_FEATS, train=tr2, mode="gate"),
        # ablations
        "AG -consistency":      dict(feats=BDT + A_ID + A_DEL, train=tr2, mode="gate_nocons"),
        "AG -delegation":       dict(feats=BDT + A_ID + A_CON, train=tr2, mode="gate_nodel"),
        "AG ML-only":           dict(feats=BDT + A_FEATS, train=tr2, mode="ml"),
        "AG -stepup(unverified)": dict(feats=BDT + A_FEATS, train=tr2, mode="gate_nounv"),
    }
    if only:
        configs = {k: v for k, v in configs.items() if k in only}
    rows, aucs, scores = [], {}, {}
    for name, cfg in configs.items():
        m = fit(cfg["train"], cfg["feats"], seed)
        thr, thr_step = threshold_for(m, cfg["feats"], cal)
        s = m.predict_proba(te[cfg["feats"]])[:, 1]
        scores[name] = s
        legit_mask = te.y == 0
        aucs[name] = dict(
            auc_all=roc_auc_score(te.y, s),
            auc_human_vs_era1=roc_auc_score(te.y[te.cls.isin(["H", "FH", "FB"])], s[te.cls.isin(["H", "FH", "FB"])]),
        )
        mode = cfg["mode"]
        if mode == "ml":
            pre = decide_ml(s, thr)
        elif mode == "allow":
            pre = np.where(te.sig_valid == 1, "approve", decide_ml(s, thr))
        else:
            tmp = te.copy()
            if mode == "gate_nocons":
                tmp["merchant_in_scope"] = 1; tmp["category_in_scope"] = 1
                tmp["amount_over_cap"] = -1; tmp["ship_to_principal"] = 1
            if mode == "gate_nodel":
                tmp["cred_bound_to_principal"] = 1
            if mode == "gate_nounv":
                tmp["webdriver"] = 0; tmp["untrusted_ratio"] = 0; tmp["headless_ua"] = 0; tmp["keydown_per_input"] = 1
            pre = apply_gate(tmp, s, thr, thr_step)
        final = resolve_stepup(te, pre, rng)
        pc = per_class(te, pre, final)
        for c, v in pc.items():
            rows.append(dict(seed=seed, policy=name, cls=c, **v))
    # per-mode false decline for legacy on legit agents (where does it bite?)
    mode_rows = []
    for name in [n for n in ["Legacy", "Retrained", "AgentGate"] if n in scores]:
        thr = np.quantile(scores[name][(te.cls == "H").values], 1 - HUMAN_DECLINE)  # approx (test humans)
        for (c, md), g in te[te.cls.isin(["AU", "AV", "FAU"])].groupby(["cls", "_mode"]):
            mode_rows.append(dict(seed=seed, policy=name, cls=c, mode=md, n=len(g),
                                  decline=float((scores[name][g.index] >= thr).mean())))
    score_dump = pd.DataFrame({"cls": te.cls, "Legacy": scores["Legacy"], "Retrained": scores["Retrained"],
                               "AgentGate": scores["AgentGate"]}) if (seed == 0 and not only) else None
    return pd.DataFrame(rows), aucs, pd.DataFrame(mode_rows), score_dump, te


CORE = ["Legacy", "Retrained", "Legacy+Allowlist", "Retrained+Allowlist", "Identity-aware", "AgentGate"]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    if which == "main":
        allr, alla, allm = [], [], []
        for sd in SEEDS:
            r, a, m, dump, te = run_seed(sd)
            allr.append(r); allm.append(m)
            alla += [dict(seed=sd, policy=k, **v) for k, v in a.items()]
            if dump is not None:
                dump.sample(60_000, random_state=0).to_csv(RES / "scores_seed0.csv", index=False)
                te.groupby("cls")[sim.B_FEATS + sim.D_FEATS].median().to_csv(RES / "feature_medians.csv")
            print("seed", sd, "done", flush=True)
        pd.concat(allr).to_csv(RES / "main.csv", index=False)
        pd.DataFrame(alla).to_csv(RES / "auc.csv", index=False)
        pd.concat(allm).to_csv(RES / "by_mode.csv", index=False)
    elif which == "fewshot":
        allr = []
        for k in [0, 25, 100, 400]:
            for sd in SEEDS[:3]:
                r, *_ = run_seed(sd, few_shot=k, only=CORE)
                r["few_shot"] = k; allr.append(r)
            print("k", k, flush=True)
        pd.concat(allr).to_csv(RES / "fewshot.csv", index=False)
    elif which == "sweep":
        allr = []
        grid = [("humanize", [0.0, 0.15, 0.4, 0.8]), ("card_binding", [0.5, 0.8, 0.95]),
                ("p_cloud_agent", [0.2, 0.7, 1.0]), ("hj_in_scope", [0.1, 0.3, 0.6])]
        for key, vals in grid:
            for v in vals:
                for sd in SEEDS[:3]:
                    r, *_ = run_seed(sd, params={key: v}, only=CORE)
                    r["param"] = key; r["value"] = v; allr.append(r)
                print(key, v, flush=True)
        pd.concat(allr).to_csv(RES / "sweep.csv", index=False)


if __name__ == "__main__":
    main()
