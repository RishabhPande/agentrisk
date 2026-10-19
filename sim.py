"""AgentRisk-Sim: semi-synthetic checkout-session generator.

Session classes
  legit:  H   human (desktop/mobile, some with browser autofill)
          AU  legitimate *unverified* agent (user-run open-source agent)
          AV  legitimate *verified* agent (Web-Bot-Auth-style signature + signed mandate)
  fraud (era 1, seen by legacy models):
          FH  manual fraudster with stolen credentials
          FB  scripted bot (card testing / account takeover)
  fraud (era 2, agent era, never in training unless stated):
          FAU fraudster driving an unverified agent
          FAV "laundering": fraudster obtains a *valid* agent identity from a lax
              platform and a mandate signed by a principal they control
          FHJ "hijack": a legitimate user's verified agent is redirected by
              prompt injection (valid identity, valid principal, off-mandate action)
          FSP spoofed agent signature (fails verification)

Agent behavioral telemetry is drawn per automation mode from distributions whose
centres come from real Playwright captures (telemetry/telemetry_summary.csv).
Everything else is parametric; every assumption is in PARAMS below.
"""
import numpy as np
import pandas as pd

LEGIT = ["H", "AU", "AV"]
ERA1_FRAUD = ["FH", "FB"]
ERA2_FRAUD = ["FAU", "FAV", "FHJ", "FSP"]
ALL = LEGIT + ERA1_FRAUD + ERA2_FRAUD

PARAMS = dict(
    p_mobile=0.40,            # share of human sessions on mobile (no mouse events)
    p_autofill=0.35,          # human desktop sessions using browser autofill
    p_vpn_human=0.04,         # humans on datacenter/VPN egress
    humanize=0.15,            # share of agent sessions with humanization/stealth jitter
    p_cloud_agent=0.70,       # agents run from cloud (datacenter IPs) vs user's machine
    card_binding=0.80,        # share of verified legit agents whose credential is bound to principal
    mandate_open=0.30,        # share of mandates that are merchant-agnostic
    legit_scope_noise=0.03,   # legit verified carts slightly over cap (tax/shipping)
    hj_in_scope=0.30,         # share of hijacks that stay inside mandate scope
)

# automation mode mix for agents and bots (modes captured in telemetry/)
AGENT_MODES = {"dom_fill": 0.40, "cdp_insert_text": 0.20, "kb_type": 0.10, "vision_cu": 0.30}
BOT_MODES = {"dom_fill": 0.35, "js_set": 0.45, "kb_type": 0.20}

# Fallback centres used only if the capture file is missing; overwritten by load_telemetry().
TELEMETRY = {
    "dom_fill":        dict(mousemove=1, kpi=0.0, iki=np.nan, straight=1.0, tts=0.3, untrusted=0.0, webdriver=1),
    "js_set":          dict(mousemove=0, kpi=0.0, iki=np.nan, straight=np.nan, tts=0.05, untrusted=1.0, webdriver=1),
    "cdp_insert_text": dict(mousemove=6, kpi=0.0, iki=np.nan, straight=1.0, tts=0.4, untrusted=0.0, webdriver=1),
    "kb_type":         dict(mousemove=6, kpi=1.0, iki=1.0, straight=1.0, tts=0.6, untrusted=0.0, webdriver=1),
    "vision_cu":       dict(mousemove=150, kpi=1.0, iki=45.0, straight=0.99, tts=9.0, untrusted=0.0, webdriver=1),
}


def load_telemetry(path):
    """Set per-mode telemetry centres from the raw Playwright capture (non-stealth runs)."""
    raw = pd.read_csv(str(path).replace("telemetry_summary.csv", "telemetry.csv"))
    raw = raw[raw.stealth == False]  # noqa: E712
    for m, g in raw.groupby("mode"):
        iki = g.iki_mean_ms.mean()
        TELEMETRY[m] = dict(
            mousemove=float(g.mousemove.mean()),
            kpi=float(g.keydown_per_input.fillna(0).mean()),
            iki=float(iki) if not np.isnan(iki) else np.nan,
            iki_cv=float((g.iki_sd_ms / g.iki_mean_ms).mean()) if not np.isnan(iki) else np.nan,
            straight=float(g.mouse_straightness.mean()) if g.mouse_straightness.notna().any() else np.nan,
            tts=float(g.time_to_submit_ms.mean()) / 1000.0,
            untrusted=float((g.untrusted / g.total).mean()),
            webdriver=float(g.webdriver.astype(float).mean()))


def _ln(rng, median, sigma, n):
    return np.exp(np.log(median) + sigma * rng.standard_normal(n))


def _behavior_human(rng, n, P, fraud=False):
    mobile = rng.random(n) < P["p_mobile"]
    autofill = (~mobile) & (rng.random(n) < P["p_autofill"])
    d = {}
    d["is_mobile"] = mobile.astype(int)
    d["mousemove"] = np.where(mobile, 0, np.round(_ln(rng, 160, 0.7, n)))
    d["touch_events"] = np.where(mobile, np.round(_ln(rng, 40, 0.6, n)), 0)
    kpi = np.clip(rng.normal(1.05, 0.12, n), 0.7, 1.6)
    kpi = np.where(autofill, rng.uniform(0.0, 0.35, n), kpi)
    d["keydown_per_input"] = kpi
    iki = _ln(rng, 190 if not fraud else 150, 0.35, n)
    d["iki_mean_ms"] = np.where(autofill & (rng.random(n) < 0.5), np.nan, iki)
    d["iki_cv"] = _ln(rng, 1.6, 0.35, n)   # incl. between-field pauses, as in the collector
    d["mouse_straightness"] = np.where(mobile, np.nan, rng.beta(4, 3, n))
    tts_med = 40 if autofill.any() else 90
    d["time_to_submit_s"] = np.where(autofill, _ln(rng, 35, 0.6, n), _ln(rng, 95, 0.6, n))
    if fraud:
        d["time_to_submit_s"] = _ln(rng, 55, 0.6, n)
    d["untrusted_ratio"] = np.zeros(n)
    d["paste_card"] = (rng.random(n) < (0.35 if fraud else 0.10)).astype(int)
    d["webdriver"] = np.zeros(n, int)
    d["headless_ua"] = np.zeros(n, int)
    return d


def _behavior_auto(rng, n, modes, humanize):
    names = list(modes)
    pick = rng.choice(names, size=n, p=np.array(list(modes.values())))
    hum = rng.random(n) < humanize
    d = {k: np.zeros(n) for k in ["mousemove", "keydown_per_input", "iki_mean_ms", "iki_cv",
                                   "mouse_straightness", "time_to_submit_s", "untrusted_ratio",
                                   "webdriver", "headless_ua"]}
    for m in names:
        idx = pick == m
        k = idx.sum()
        if k == 0:
            continue
        t = TELEMETRY[m]
        d["mousemove"][idx] = np.round(np.maximum(0, t["mousemove"] * _ln(rng, 1, 0.3, k)))
        d["keydown_per_input"][idx] = np.clip(t["kpi"] + rng.normal(0, 0.02, k), 0, 2)
        d["iki_mean_ms"][idx] = np.nan if np.isnan(t["iki"]) else t["iki"] * _ln(rng, 1, 0.15, k)
        d["iki_cv"][idx] = np.nan if np.isnan(t["iki"]) else t.get("iki_cv", 0.05) * _ln(rng, 1, 0.2, k)
        d["mouse_straightness"][idx] = np.nan if np.isnan(t["straight"]) else np.clip(t["straight"] + rng.normal(0, 0.005, k), 0, 1)
        # LLM agents add model latency (planning steps) on top of primitive execution time
        d["time_to_submit_s"][idx] = t["tts"] + _ln(rng, 25, 0.7, k)
        d["untrusted_ratio"][idx] = t["untrusted"]
        d["webdriver"][idx] = t["webdriver"]
        d["headless_ua"][idx] = 1.0
    # humanized / stealth agents: jittered typing, curved mouse paths, patched webdriver flag
    k = hum.sum()
    d["mousemove"][hum] = np.round(_ln(rng, 120, 0.6, k))
    d["keydown_per_input"][hum] = np.clip(rng.normal(1.0, 0.05, k), 0.8, 1.2)
    d["iki_mean_ms"][hum] = _ln(rng, 140, 0.25, k)
    d["iki_cv"][hum] = _ln(rng, 1.2, 0.35, k)
    d["mouse_straightness"][hum] = rng.beta(8, 2, k)
    d["webdriver"][hum] = 0
    d["headless_ua"][hum] = 0
    d["is_mobile"] = np.zeros(n, int)
    d["touch_events"] = np.zeros(n)
    d["paste_card"] = np.zeros(n, int)
    d["_mode"] = np.where(hum, "humanized", pick)
    return d


def _txn(rng, n, kind):
    """Transaction/account features. kind in {legit, legit_agent, fraud, bot, hijack}."""
    d = {}
    if kind in ("legit", "legit_agent", "hijack"):
        guest = rng.random(n) < (0.15 if kind == "legit" else 0.45)
        d["account_age_d"] = np.where(guest, 0, _ln(rng, 420, 1.0, n))
        d["amount"] = _ln(rng, 60, 0.9, n)
        d["bill_ship_mismatch"] = (rng.random(n) < 0.08).astype(int)
        d["bin_country_mismatch"] = (rng.random(n) < 0.03).astype(int)
        d["cards_on_device_24h"] = 1 + rng.poisson(0.05, n)
        d["attempts_1h"] = 1 + rng.poisson(0.10 if kind == "legit" else 0.6, n)  # agents retry
        d["new_ship_addr"] = (rng.random(n) < 0.07).astype(int)
        d["high_risk_item"] = (rng.random(n) < 0.12).astype(int)
        if kind == "hijack":   # attacker redirects goods / buys resellable items
            d["new_ship_addr"] = (rng.random(n) < 0.35).astype(int)
            d["bill_ship_mismatch"] = (rng.random(n) < 0.30).astype(int)
            d["high_risk_item"] = (rng.random(n) < 0.30).astype(int)
            d["amount"] = _ln(rng, 90, 0.9, n)
    elif kind == "fraud":
        new = rng.random(n) < 0.40
        d["account_age_d"] = np.where(new, rng.uniform(0, 3, n), _ln(rng, 300, 1.2, n))  # ATO keeps age
        d["amount"] = _ln(rng, 95, 1.0, n)
        d["bill_ship_mismatch"] = (rng.random(n) < 0.22).astype(int)
        d["bin_country_mismatch"] = (rng.random(n) < 0.10).astype(int)
        d["cards_on_device_24h"] = 1 + rng.poisson(0.35, n)
        d["attempts_1h"] = 1 + rng.poisson(0.45, n)
        d["new_ship_addr"] = (rng.random(n) < 0.25).astype(int)
        d["high_risk_item"] = (rng.random(n) < 0.28).astype(int)
    elif kind == "bot":
        d["account_age_d"] = np.where(rng.random(n) < 0.5, 0, _ln(rng, 300, 1.0, n))
        testing = rng.random(n) < 0.35
        d["amount"] = np.where(testing, rng.uniform(0.5, 5, n), _ln(rng, 150, 0.8, n))
        d["bill_ship_mismatch"] = (rng.random(n) < 0.45).astype(int)
        d["bin_country_mismatch"] = (rng.random(n) < 0.35).astype(int)
        d["cards_on_device_24h"] = np.where(testing, 3 + rng.poisson(6, n), 1 + rng.poisson(0.8, n))
        d["attempts_1h"] = np.where(testing, 3 + rng.poisson(8, n), 1 + rng.poisson(1.5, n))
        d["new_ship_addr"] = (rng.random(n) < 0.50).astype(int)
        d["high_risk_item"] = (rng.random(n) < 0.50).astype(int)
    return d


def _network(rng, n, kind, P):
    d = {}
    if kind == "human":
        dc = rng.random(n) < P["p_vpn_human"]
        d["datacenter_ip"] = dc.astype(int)
        d["ip_sessions_24h"] = np.round(1 + _ln(rng, 2, 0.8, n) + dc * _ln(rng, 30, 1, n))
    elif kind == "agent":
        dc = rng.random(n) < P["p_cloud_agent"]   # cloud agents share egress IPs across many users
        d["datacenter_ip"] = dc.astype(int)
        d["ip_sessions_24h"] = np.round(np.where(dc, _ln(rng, 800, 1.0, n), 1 + _ln(rng, 3, 0.8, n)))
    elif kind == "fraud_human":
        dc = rng.random(n) < 0.20
        d["datacenter_ip"] = dc.astype(int)
        d["ip_sessions_24h"] = np.round(1 + _ln(rng, 3, 1.0, n) + dc * _ln(rng, 30, 1, n))
    elif kind == "bot":
        dc = rng.random(n) < 0.75
        d["datacenter_ip"] = dc.astype(int)
        d["ip_sessions_24h"] = np.round(_ln(rng, 300, 1.2, n))
    return d


def _agent_layer(rng, n, kind, P):
    """Agent-aware features (the proposed signals). NaN-free; 0 when absent."""
    z = np.zeros(n)
    d = dict(sig_present=z.copy(), sig_valid=z.copy(), platform_kyc_tier=z.copy() - 1,
             mandate_present=z.copy(), mandate_valid=z.copy(), principal_age_d=z.copy() - 1,
             cred_bound_to_principal=z.copy(), mandate_age_min=z.copy() - 1, chain_depth=z.copy(),
             amount_over_cap=z.copy(), merchant_in_scope=np.ones(n), category_in_scope=np.ones(n),
             ship_to_principal=np.ones(n), human_present=z.copy())
    if kind in ("AV", "FHJ"):
        d["sig_present"][:] = 1; d["sig_valid"][:] = 1
        d["platform_kyc_tier"] = rng.choice([0, 1, 2], n, p=[0.05, 0.25, 0.70]).astype(float)
        d["mandate_present"][:] = 1; d["mandate_valid"][:] = 1
        d["principal_age_d"] = _ln(rng, 600, 0.9, n)
        d["cred_bound_to_principal"] = (rng.random(n) < P["card_binding"]).astype(float)
        hp = rng.random(n) < 0.6
        d["human_present"] = hp.astype(float)
        d["mandate_age_min"] = np.where(hp, _ln(rng, 3, 0.8, n), _ln(rng, 60 * 24 * 3, 1.0, n))
        d["chain_depth"] = np.where(rng.random(n) < 0.8, 1, 2).astype(float)
        if kind == "AV":
            # benign scope deviations: taxes/shipping push a few carts slightly over cap
            noise = rng.random(n) < P["legit_scope_noise"]
            d["amount_over_cap"] = np.where(noise, rng.uniform(0.0, 0.08, n), -rng.uniform(0.05, 0.9, n))
            d["category_in_scope"] = (rng.random(n) > 0.01).astype(float)
        else:  # hijack: some stay in scope, most violate at least one constraint
            in_scope = rng.random(n) < P["hj_in_scope"]
            open_m = rng.random(n) < P["mandate_open"]
            viol = rng.choice(["merchant", "category", "amount"], n, p=[0.4, 0.3, 0.3])
            d["merchant_in_scope"] = np.where(in_scope | open_m | (viol != "merchant"), 1, 0).astype(float)
            d["category_in_scope"] = np.where(in_scope | (viol != "category"), 1, 0).astype(float)
            d["amount_over_cap"] = np.where(~in_scope & (viol == "amount"), rng.uniform(0.2, 3.0, n),
                                            -rng.uniform(0.0, 0.3, n))
            # in-scope hijacks (e.g. attacker-listed product under an open mandate) mostly keep the
            # principal's address; out-of-scope hijacks usually redirect delivery
            d["ship_to_principal"] = np.where(in_scope, rng.random(n) < 0.8, rng.random(n) < 0.35).astype(float)
    elif kind == "FAV":
        d["sig_present"][:] = 1; d["sig_valid"][:] = 1
        d["platform_kyc_tier"] = rng.choice([0, 1, 2], n, p=[0.60, 0.30, 0.10]).astype(float)
        d["mandate_present"][:] = 1; d["mandate_valid"][:] = 1   # self-signed by attacker-controlled principal
        d["principal_age_d"] = rng.uniform(0, 5, n)
        d["cred_bound_to_principal"] = (rng.random(n) < 0.05).astype(float)  # stolen card not bound
        d["human_present"] = (rng.random(n) < 0.5).astype(float)
        d["mandate_age_min"] = _ln(rng, 2, 0.8, n)
        d["chain_depth"] = np.ones(n)
        d["amount_over_cap"] = -rng.uniform(0.0, 0.2, n)          # attacker sets cap to fit
        d["ship_to_principal"] = (rng.random(n) < 0.9).astype(float)  # principal = attacker
    elif kind == "FSP":
        d["sig_present"][:] = 1; d["sig_valid"][:] = 0
        d["platform_kyc_tier"][:] = -1
    return d


def generate(n_by_class, seed=0, params=None):
    P = dict(PARAMS); P.update(params or {})
    rng = np.random.default_rng(seed)
    frames = []
    for c, n in n_by_class.items():
        if n <= 0:
            continue
        if c == "H":
            d = {**_behavior_human(rng, n, P), **_txn(rng, n, "legit"), **_network(rng, n, "human", P)}
        elif c in ("AU", "AV"):
            d = {**_behavior_auto(rng, n, AGENT_MODES, P["humanize"]), **_txn(rng, n, "legit_agent"),
                 **_network(rng, n, "agent", P)}
        elif c == "FH":
            d = {**_behavior_human(rng, n, P, fraud=True), **_txn(rng, n, "fraud"), **_network(rng, n, "fraud_human", P)}
        elif c == "FB":
            d = {**_behavior_auto(rng, n, BOT_MODES, 0.10), **_txn(rng, n, "bot"), **_network(rng, n, "bot", P)}
        elif c in ("FAU", "FAV", "FSP"):
            d = {**_behavior_auto(rng, n, AGENT_MODES, P["humanize"]), **_txn(rng, n, "fraud"),
                 **_network(rng, n, "agent", P)}
        elif c == "FHJ":
            d = {**_behavior_auto(rng, n, AGENT_MODES, P["humanize"]), **_txn(rng, n, "hijack"),
                 **_network(rng, n, "agent", P)}
        d.update(_agent_layer(rng, n, c, P))
        df = pd.DataFrame(d)
        df["cls"] = c
        df["y"] = int(c not in LEGIT)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "_mode" not in out:
        out["_mode"] = "human"
    out["_mode"] = out["_mode"].fillna("human")
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


B_FEATS = ["is_mobile", "mousemove", "touch_events", "keydown_per_input", "iki_mean_ms", "iki_cv",
           "mouse_straightness", "time_to_submit_s", "untrusted_ratio", "paste_card"]
D_FEATS = ["webdriver", "headless_ua", "datacenter_ip", "ip_sessions_24h"]
T_FEATS = ["account_age_d", "amount", "bill_ship_mismatch", "bin_country_mismatch",
           "cards_on_device_24h", "attempts_1h", "new_ship_addr", "high_risk_item"]
A_ID = ["sig_present", "sig_valid", "platform_kyc_tier"]
A_DEL = ["mandate_present", "mandate_valid", "principal_age_d", "cred_bound_to_principal",
         "mandate_age_min", "chain_depth", "human_present"]
A_CON = ["amount_over_cap", "merchant_in_scope", "category_in_scope", "ship_to_principal"]
A_FEATS = A_ID + A_DEL + A_CON
