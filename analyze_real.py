"""Tables and macros for the real production-agent validation (paper/gen/real_*.tex)."""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent
GEN = ROOT / "paper" / "gen"
real = pd.read_csv(ROOT / "live" / "real_agent_sessions.csv")
real["mode"] = (real.keydown > 0).map({True: "keyboard", False: "click"})
frames = [pd.read_csv(ROOT / "results" / f"real_agents_{c}_cloud{k}.csv") for c in ("AU", "AV") for k in (0, 1)]
d = pd.concat(frames)
d["is_agent"] = d["mode"] != "human (simulated)"
agg = d.groupby(["ctx", "cloud", "policy", "is_agent"]).agg(dec=("decline", "mean"), step=("stepup", "mean")).reset_index()

macros = {}
def mac(k, v, fmt="{:.1f}"):
    macros[k] = fmt.format(v) if not isinstance(v, str) else v

cols = [("AU", 0.0), ("AU", 1.0), ("AV", 0.0), ("AV", 1.0)]
names = {"Legacy": "Legacy", "Retrained": "Retrained", "AgentGate": "\\textbf{AgentGate}"}
lines = []
for p in ["Legacy", "Retrained", "AgentGate"]:
    cells = []
    for ctx, cl in cols:
        a = agg[(agg.ctx == ctx) & (agg.cloud == cl) & (agg.policy == p)]
        ag = 100 * a[a.is_agent].dec.iloc[0]
        hu = 100 * a[~a.is_agent].dec.iloc[0]
        cells.append(f"{ag:.1f} & \\textcolor{{black!55}}{{{hu:.1f}}}")
        key = f"real{p}{ctx}{'Cloud' if cl else 'Local'}"
        mac(key, ag); mac(key + "Hum", hu)
        mac(key + "Step", 100 * a[a.is_agent].step.iloc[0])
    lines.append(names[p] + " & " + " & ".join(cells) + " \\\\")
(GEN / "tab_real.tex").write_text("\n".join(lines) + "\n")

# per-mode legacy numbers (local, unverified) for the text
m = d[(d.ctx == "AU") & (d.cloud == 0.0) & (d.policy == "Legacy")].groupby("mode").decline.mean() * 100
mac("realLegClickLocal", m["click"]); mac("realLegKbLocal", m["keyboard"])
m = d[(d.ctx == "AU") & (d.cloud == 1.0) & (d.policy == "Legacy")].groupby("mode").decline.mean() * 100
mac("realLegClickCloud", m["click"]); mac("realLegKbCloud", m["keyboard"])

mac("realN", str(len(real)))
mac("realNclick", str(int((real["mode"] == "click").sum())))
mac("realNkb", str(int((real["mode"] == "keyboard").sum())))
mac("realClicks", str(int(real[real["mode"] == "click"].click.iloc[0])))
mac("realTTSmin", real.time_to_submit_ms.min() / 1000, "{:.1f}")
mac("realTTSmax", real.time_to_submit_ms.max() / 1000, "{:.1f}")
mac("realIKIkb", real[real["mode"] == "keyboard"].iki_mean_ms.min(), "{:.1f}")
assert (real.mousemove == 0).all() and (~real.webdriver).all() and (~real.ua_headless).all()

# telemetry table rows for the real sessions
rows = []
for md, lab in [("click", "Claude computer use, click mode"), ("keyboard", "Claude computer use, keyboard mode")]:
    g = real[real["mode"] == md]
    rows.append(f"{lab} ($n{{=}}{len(g)}$) & {g.mousemove.mean():.0f} & {g.mousedown.mean():.0f} & "
                f"{g.keydown.mean():.0f}/{g.input.mean():.0f} & {g.untrusted.sum():.0f} & "
                f"{g.time_to_submit_ms.median()/1000:.1f} & {int(g.webdriver.any())} & {int(g.ua_headless.any())} \\\\")
(GEN / "tab_real_tel.tex").write_text("\n".join(rows) + "\n")

with open(GEN / "numbers_real.tex", "w") as f:
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(macros), "macros"); print((GEN / "tab_real.tex").read_text()); print((GEN / "tab_real_tel.tex").read_text())
