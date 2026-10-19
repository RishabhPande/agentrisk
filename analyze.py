"""Turn results/*.csv into figures and LaTeX macros/tables (paper/gen/)."""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
RES = ROOT / "results"
GEN = ROOT / "paper" / "gen"; GEN.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "paper" / "figs"; FIG.mkdir(parents=True, exist_ok=True)

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})

POL_ORDER = ["Legacy", "Legacy+Allowlist", "Retrained", "Retrained+Allowlist", "Identity-aware", "AgentGate"]
POL_COLOR = {"Legacy": INK2, "Legacy+Allowlist": C["yellow"], "Retrained": C["orange"],
             "Retrained+Allowlist": C["magenta"], "Identity-aware": C["violet"], "AgentGate": C["blue"]}
CLS_ORDER = ["H", "AU", "AV", "FH", "FB", "FAU", "FAV", "FHJ", "FSP"]
ERA2 = ["FAU", "FAV", "FHJ", "FSP"]
macros = {}


def mac(name, val, fmt="{:.1f}"):
    macros[name] = fmt.format(val) if not isinstance(val, str) else val


def pct(x):
    return 100 * x


# ---------------- main results ----------------
main = pd.read_csv(RES / "main.csv")
agg = main.groupby(["policy", "cls"]).agg(dec=("decline", "mean"), dec_sd=("decline", "std"),
                                          step=("stepup", "mean")).reset_index()
piv = agg.pivot(index="policy", columns="cls", values="dec")
pivsd = agg.pivot(index="policy", columns="cls", values="dec_sd")
pivstep = agg.pivot(index="policy", columns="cls", values="step")

names_tex = {"Legacy": "Legacy", "Legacy+Allowlist": "Legacy + allowlist", "Retrained": "Retrained",
             "Retrained+Allowlist": "Retrained + allowlist", "Identity-aware": "Identity-aware ML",
             "AgentGate": "\\textbf{AgentGate (ours)}", "AgentGate-invariant": "AgentGate, automation-invariant",
             "AG -consistency": "\\quad $-$ scope (consistency) checks", "AG -delegation": "\\quad $-$ credential binding",
             "AG ML-only": "\\quad $-$ all gates (ML only)", "AG -stepup(unverified)": "\\quad $-$ step-up for unattested automation"}


def table(policies, path, caption_note=""):
    lines = []
    for p in policies:
        cells = []
        for c in CLS_ORDER:
            v, sd = pct(piv.loc[p, c]), pct(pivsd.loc[p, c])
            s = f"{v:.1f}"
            if c in ("AU", "AV") and v > 20:
                s = f"\\cellcolor{{bad}}{s}"
            if c in ERA2 + ["FH", "FB"] and v < 50:
                s = f"\\cellcolor{{bad}}{s}"
            cells.append(s)
        lines.append(names_tex[p] + " & " + " & ".join(cells) + " \\\\")
    (GEN / path).write_text("\n".join(lines) + "\n")


table(POL_ORDER, "tab_main.tex")
ABL = ["AgentGate", "AG ML-only", "AG -consistency", "AG -delegation", "AG -stepup(unverified)", "AgentGate-invariant"]
table(ABL, "tab_ablation.tex")

# macros for text
for p, key in [("Legacy", "Leg"), ("Retrained", "Ret"), ("Legacy+Allowlist", "LegAl"), ("Retrained+Allowlist", "RetAl"),
               ("Identity-aware", "Id"), ("AgentGate", "AG"), ("AG -consistency", "AGnoCon"), ("AG -delegation", "AGnoDel"),
               ("AG ML-only", "AGml"), ("AG -stepup(unverified)", "AGnoUnv"), ("AgentGate-invariant", "AGinv")]:
    for c in CLS_ORDER:
        mac(f"{key}{c}", pct(piv.loc[p, c]))
        mac(f"{key}{c}sd", pct(pivsd.loc[p, c]))
    mac(f"{key}AgentFD", pct((piv.loc[p, "AU"] + piv.loc[p, "AV"]) / 2))
    mac(f"{key}EraTwo", pct(piv.loc[p, ERA2].mean()))
    mac(f"{key}StepAV", pct(pivstep.loc[p, "AV"]))
    mac(f"{key}StepAU", pct(pivstep.loc[p, "AU"]))
    mac(f"{key}StepH", pct(pivstep.loc[p, "H"]), "{:.2f}")
mac("FADratioLeg", piv.loc["Legacy", "AV"] / piv.loc["Legacy", "H"], "{:.0f}")
mac("FADratioAG", piv.loc["AgentGate", "AV"] / piv.loc["AgentGate", "H"], "{:.1f}")
mac("SafeHarborDrop", pct(piv.loc["Legacy", "FAU"] - piv.loc["Retrained", "FAU"]))
mac("nSeeds", str(main.seed.nunique()))

auc = pd.read_csv(RES / "auc.csv").groupby("policy").mean(numeric_only=True)
for p, key in [("Legacy", "Leg"), ("Retrained", "Ret"), ("AgentGate", "AG")]:
    mac(f"{key}AUCeraone", auc.loc[p, "auc_human_vs_era1"], "{:.3f}")

# ---------------- Fig 1: score distributions of the legacy model ----------------
sc = pd.read_csv(RES / "scores_seed0.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
groups = [("H", "Legit human", INK2, "-"), ("AV", "Legit agent (verified)", C["blue"], "-"),
          ("AU", "Legit agent (unverified)", C["aqua"], "--"), ("FH", "Fraud: manual", C["orange"], "-"),
          ("FB", "Fraud: scripted bot", C["red"], "--")]
for ax, model, title in [(axes[0], "Legacy", "(a) Legacy model (pre-agent training data)"),
                         (axes[1], "AgentGate", "(b) AgentGate scorer")]:
    thr = np.quantile(sc.loc[sc.cls == "H", model], 0.98)
    for c, lab, col, ls in groups:
        s = np.sort(np.clip(sc.loc[sc.cls == c, model].values, 1e-6, 1 - 1e-6))
        lo = np.log10(s / (1 - s))
        ax.plot(lo, np.arange(1, len(lo) + 1) / len(lo), color=col, lw=1.6, ls=ls, label=lab)
    t = np.log10(thr / (1 - thr))
    ax.axvline(t, color=INK, lw=0.8, ls=":")
    ax.text(t + 0.1, 0.04, "decline\nthreshold", fontsize=7, color=INK2, va="bottom")
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.set_xlabel("risk score (log$_{10}$ odds)")
    ax.set_xlim(-6.5, 5.5)
axes[0].set_ylabel("cumulative share of sessions")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.28), ncol=5, fontsize=7, handlelength=2.2)
fig.savefig(FIG / "fig_scores.pdf")
plt.close(fig)

# ---------------- Fig 2: failure-mode map ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.7))
offs = {"Legacy": (-8, 0), "Legacy+Allowlist": (-6, 9), "Retrained": (7, 0), "Retrained+Allowlist": (7, 7),
        "Identity-aware": (7, -7), "AgentGate": (7, 0)}
for p in POL_ORDER:
    x = pct((piv.loc[p, "AU"] + piv.loc[p, "AV"]) / 2)
    y = pct(piv.loc[p, ERA2].mean())
    ax.scatter(x, y, s=46, color=POL_COLOR[p], edgecolor="white", linewidth=1.5, zorder=3)
    ha = "right" if p in ("Legacy", "Legacy+Allowlist") else "left"
    ax.annotate(p.replace("+", " + "), (x, y), xytext=offs[p], textcoords="offset points", fontsize=7,
                color=INK, ha=ha, va="center")
ax.set_xlabel("false declines on legitimate agents (%)")
ax.set_ylabel("agent-era fraud stopped (%)")
ax.set_xlim(-5, 105); ax.set_ylim(0, 105)
ax.scatter(0, 100, marker="*", s=60, color="none", edgecolor=INK2, linewidth=0.8, zorder=2)
ax.annotate("ideal", (0, 100), xytext=(7, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
fig.savefig(FIG / "fig_map.pdf")
plt.close(fig)

# ---------------- Fig 3: per-attack bars for the key policies ----------------
fig, ax = plt.subplots(figsize=(7.0, 2.1))
pols = ["Legacy", "Retrained", "Retrained+Allowlist", "Identity-aware", "AgentGate"]
cls_lab = {"FH": "manual\nfraud", "FB": "scripted\nbot", "FAU": "fraud via\nunverif. agent",
           "FAV": "laundering\n(verif. agent)", "FHJ": "hijacked\nagent", "FSP": "spoofed\nsignature",
           "AU": "legit agent\n(unverified)", "AV": "legit agent\n(verified)"}
cl = ["AV", "AU", "FH", "FB", "FAU", "FAV", "FHJ", "FSP"]
w = 0.16
for i, p in enumerate(pols):
    xs = np.arange(len(cl)) + (i - 2) * w
    ys = [pct(piv.loc[p, c]) for c in cl]
    ax.bar(xs, ys, width=w - 0.02, color=POL_COLOR[p], label=p.replace("+", " + "), zorder=3)
ax.axvline(1.5, color=INK2, lw=0.6)
ax.text(0.5, 103, "legit: lower is better", ha="center", fontsize=7, color=INK2)
ax.text(4.5, 103, "fraud: higher is better", ha="center", fontsize=7, color=INK2)
ax.set_xticks(range(len(cl))); ax.set_xticklabels([cls_lab[c] for c in cl], fontsize=7)
ax.set_ylabel("declined (%)"); ax.set_ylim(0, 110)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=7)
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig_bars.pdf")
plt.close(fig)

# ---------------- by automation mode ----------------
bm = pd.read_csv(RES / "by_mode.csv")
bm = bm.groupby(["policy", "cls", "mode"]).decline.mean().reset_index()
legit_modes = bm[bm.cls.isin(["AU", "AV"])].groupby(["policy", "mode"]).decline.mean().unstack(0)
for md in legit_modes.index:
    key = md.replace("_", "")
    mac(f"LegMode{key}", pct(legit_modes.loc[md, "Legacy"]))
    mac(f"AGMode{key}", pct(legit_modes.loc[md, "AgentGate"]))

# ---------------- telemetry table ----------------
tel = pd.read_csv(ROOT / "telemetry" / "telemetry.csv")
tel["untrusted_frac"] = tel.untrusted / tel.total
g = tel.groupby(["mode", "stealth"]).agg(mm=("mousemove", "mean"), kd=("keydown", "mean"), inp=("input", "mean"),
                                         unt=("untrusted_frac", "mean"), iki=("iki_mean_ms", "mean"),
                                         tts=("time_to_submit_ms", "median"), wd=("webdriver", "mean"),
                                         hl=("ua_headless", "mean"), n=("rep", "count")).reset_index()
mode_tex = {"dom_fill": "DOM \\texttt{fill}", "js_set": "JS value + \\texttt{dispatchEvent}",
            "cdp_insert_text": "CDP \\texttt{insertText}", "kb_type": "keystrokes, no delay",
            "kb_type_delay100": "keystrokes, 100\\,ms delay", "vision_cu": "computer-use style"}
lines = []
for md in ["dom_fill", "js_set", "cdp_insert_text", "kb_type", "kb_type_delay100", "vision_cu"]:
    a = g[(g["mode"] == md) & (g.stealth == False)].iloc[0]  # noqa: E712
    b = g[(g["mode"] == md) & (g.stealth == True)].iloc[0]  # noqa: E712
    iki = "--" if np.isnan(a.iki) else f"{a.iki:.1f}"
    lines.append(f"{mode_tex[md]} & {a.mm:.0f} & {a.kd:.0f}/{a.inp:.0f} & {a.unt:.2f} & {iki} & "
                 f"{a.tts/1000:.2f} & {a.wd:.0f}/{b.wd:.0f} & {a.hl:.0f}/{b.hl:.0f} \\\\")
(GEN / "tab_telemetry.tex").write_text("\n".join(lines) + "\n")
mac("telReps", str(int(g.n.min())))
mac("telIKIkb", g[(g["mode"] == "kb_type") & (g.stealth == False)].iki.iloc[0], "{:.1f}")  # noqa: E712
mac("telIKIcua", g[(g["mode"] == "vision_cu") & (g.stealth == False)].iki.iloc[0], "{:.0f}")  # noqa: E712

# ---------------- cost model ----------------
# Expected loss per 10k checkout sessions; parameters stated in the paper.
AMT = dict(H=60 * np.exp(0.9**2 / 2), AU=60 * np.exp(0.9**2 / 2), AV=60 * np.exp(0.9**2 / 2))
MARGIN, CHURN, FRAUD_AMT, CB_FEE, STEP_COST = 0.30, 10.0, 95 * np.exp(1.0**2 / 2), 20.0, 0.25


def cost(p, agent_share, fraud_rate=0.01, era2_share=0.5):
    legit = 1 - fraud_rate
    mix = {"H": legit * (1 - agent_share), "AU": legit * agent_share * 0.5, "AV": legit * agent_share * 0.5}
    f1 = fraud_rate * (1 - era2_share); f2 = fraud_rate * era2_share
    mix.update({"FH": f1 * 0.6, "FB": f1 * 0.4, "FAU": f2 * 0.4, "FAV": f2 * 0.2, "FHJ": f2 * 0.3, "FSP": f2 * 0.1})
    fd = sum(mix[c] * piv.loc[p, c] * (AMT[c] * MARGIN + CHURN) for c in ["H", "AU", "AV"])
    fr = sum(mix[c] * (1 - piv.loc[p, c]) * (FRAUD_AMT + CB_FEE) for c in ["FH", "FB", "FAU", "FAV", "FHJ", "FSP"])
    st = sum(mix[c] * pivstep.loc[p, c] * STEP_COST for c in ["H", "AU", "AV"])
    return 10_000 * fd, 10_000 * fr, 10_000 * st


rows = []
for share in [0.05, 0.20, 0.40]:
    for p in POL_ORDER:
        fd, fr, st = cost(p, share)
        rows.append(dict(share=share, policy=p, fd=fd, fr=fr, st=st, tot=fd + fr + st))
cdf = pd.DataFrame(rows)
cdf.to_csv(RES / "cost.csv", index=False)
lines = []
for p in POL_ORDER:
    r = cdf[cdf.policy == p].set_index("share")
    cells = []
    for share in [0.05, 0.20, 0.40]:
        x = r.loc[share]
        cells.append(f"{x.fd:,.0f} & {x.fr:,.0f} & {x.tot:,.0f}".replace(",", "{,}"))
    lines.append(names_tex[p] + " & " + " & ".join(cells) + " \\\\")
(GEN / "tab_cost.tex").write_text("\n".join(lines) + "\n")
for share, k in [(0.05, "five"), (0.20, "twenty"), (0.40, "forty")]:
    r = cdf[cdf.share == share].set_index("policy").tot
    mac(f"costLeg{k}", r["Legacy"], "{:,.0f}")
    mac(f"costAG{k}", r["AgentGate"], "{:,.0f}")
    mac(f"costRet{k}", r["Retrained"], "{:,.0f}")
    mac(f"costRedAG{k}", pct(1 - r["AgentGate"] / r["Legacy"]), "{:.0f}")
    best_other = r.drop("AgentGate").min()
    mac(f"costBestOther{k}", best_other, "{:,.0f}")
    mac(f"costBestOtherName{k}", r.drop("AgentGate").idxmin())

# ---------------- few-shot ----------------
fs_path = RES / "fewshot.csv"
if fs_path.exists():
    fs = pd.read_csv(fs_path).groupby(["few_shot", "policy", "cls"]).decline.mean().unstack("cls")
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
    for ax, c in zip(axes, ["FAU", "FAV", "FHJ", "AGENT"]):
        for p in ["Retrained", "Identity-aware", "AgentGate"]:
            dd = fs.xs(p, level="policy")
            d = (dd["AU"] + dd["AV"]) / 2 if c == "AGENT" else dd[c]
            ax.plot(range(len(d)), pct(d.values), marker="o", ms=4, lw=1.6, color=POL_COLOR[p], label=p,
                    ls="--" if c == "AGENT" else "-")
        ax.set_xticks(range(len(d))); ax.set_xticklabels(d.index)
        ttl = {"FAU": "FAU stopped", "FAV": "FAV (laundering) stopped", "FHJ": "FHJ (hijack) stopped",
               "AGENT": "legit agents declined"}[c]
        ax.set_title(ttl, fontsize=7.5, loc="left")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("%")
    fig.supxlabel("labeled examples per agent-era attack class", fontsize=7.5, y=-0.10)
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(2.3, -0.30), ncol=3)
    fig.savefig(FIG / "fig_fewshot.pdf")
    plt.close(fig)
    for p, key in [("Retrained", "Ret"), ("Identity-aware", "Id"), ("AgentGate", "AG")]:
        for c in ["FAU", "FAV", "FHJ"]:
            mac(f"fs{key}{c}max", pct(fs.loc[(fs.index.get_level_values(0).max(), p), c]))
            mac(f"fs{key}{c}zero", pct(fs.loc[(0, p), c]))
    mac("fsMax", str(fs.index.get_level_values(0).max()))
    for p, key in [("AgentGate", "AG"), ("Retrained", "Ret"), ("Identity-aware", "Id")]:
        for c in ["AU", "AV", "H"]:
            mac(f"fs{key}{c}max", pct(fs.loc[(fs.index.get_level_values(0).max(), p), c]))

# ---------------- sensitivity ----------------
sw_path = RES / "sweep.csv"
if sw_path.exists():
    sw = pd.read_csv(sw_path).groupby(["param", "value", "policy", "cls"]).decline.mean().reset_index()
    titles = {"humanize": "humanized agents", "card_binding": "credential binding",
              "p_cloud_agent": "cloud-hosted agents", "hj_in_scope": "in-scope hijacks"}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for ax, prm in zip(axes, ["humanize", "card_binding", "p_cloud_agent", "hj_in_scope"]):
        d = sw[sw.param == prm]
        for p, metric, ls, lab in [("Legacy", "agent", "-", "Legacy: legit-agent FD"),
                                   ("AgentGate", "agent", "-", "AgentGate: legit-agent FD"),
                                   ("AgentGate", "era2", "--", "AgentGate: agent-era fraud stopped"),
                                   ("Retrained", "era2", "--", "Retrained: agent-era fraud stopped")]:
            dd = d[d.policy == p]
            if metric == "agent":
                y = dd[dd.cls.isin(["AU", "AV"])].groupby("value").decline.mean()
            else:
                y = dd[dd.cls.isin(ERA2)].groupby("value").decline.mean()
            ax.plot(y.index, pct(y.values), marker="o", ms=3.5, lw=1.5, ls=ls, color=POL_COLOR[p], label=lab)
        ax.set_title(titles[prm], fontsize=7.5, loc="left")
        ax.set_ylim(0, 105)
        ax.set_xlabel("share", fontsize=7)
    axes[0].set_ylabel("rate (%)")
    axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(2.3, -0.38), ncol=2)
    fig.savefig(FIG / "fig_sweep.pdf")
    plt.close(fig)
    s2 = sw.copy()
    agfd = s2[(s2.policy == "AgentGate") & s2.cls.isin(["AU", "AV"])].groupby(["param", "value"]).decline.mean()
    ag2 = s2[(s2.policy == "AgentGate") & s2.cls.isin(ERA2)].groupby(["param", "value"]).decline.mean()
    legfd = s2[(s2.policy == "Legacy") & s2.cls.isin(["AU", "AV"])].groupby(["param", "value"]).decline.mean()
    mac("swAGfdMax", pct(agfd.max())); mac("swAGeraTwoMin", pct(ag2.min()))
    mac("swLegfdMin", pct(legfd.min()))
    mac("swLegfdHumMax", pct(legfd.loc["humanize"].min()))
    mac("swAGfavBindLow", pct(s2[(s2.policy == "AgentGate") & (s2.cls == "AV") & (s2.param == "card_binding")]
                              .groupby("value").decline.mean().loc[0.5]))
    hj = s2[(s2.policy == "AgentGate") & (s2.cls == "FHJ") & (s2.param == "hj_in_scope")].groupby("value").decline.mean()
    mac("swAGhjLow", pct(hj.iloc[0])); mac("swAGhjHigh", pct(hj.iloc[-1]))

with open(GEN / "numbers.tex", "w") as f:
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k.replace('-', '').replace(' ', '')}}}{{{v}}}\n")
print(f"{len(macros)} macros written")
print((piv[CLS_ORDER] * 100).round(1).loc[POL_ORDER + ABL[1:]].to_string())
print(cdf.pivot(index="policy", columns="share", values="tot").round(0))
