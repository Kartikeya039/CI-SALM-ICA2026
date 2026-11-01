"""CI-SALM: Climate-Integrated Stochastic Asset-Liability Model.

Maps NGFS Phase V transition scenarios into a stochastic pension ALM and
compares it with a stationary (traditional) ALM on identical random shocks.

    python cisalm.py              # all results, printed and saved to results.json
    python cisalm.py --download   # refresh data/ngfs_phase5.csv from IIASA (needs pyam-iamc)
"""
import argparse, copy, json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "ngfs_phase5.csv"
MODEL = "MESSAGEix-GLOBIOM 2.0-M-R12-NGFS"
SCENARIOS = {"orderly": "Below 2°C", "disorderly": "Delayed transition"}
N, T, START = 10_000, 30, 2025
YEARS = np.arange(START, START + T + 1)

FUND = dict(w_eq=0.60, w_bd=0.40, fossil_share=0.10, D_B=7.0, D_L=14.0,
            accrual_excess=0.006, liab_noise=0.010, idx_wage=0.50, credit_share=0.50)
MARKET = dict(y0=0.038, y_bar=0.038, kappa=0.15, sigma_y=0.008, erp=0.030,
              sigma_e=0.16, bond_noise=0.0)
COEF = dict(theta_E=0.30, profit_share=0.10,             # carbon-cost incidence on equity
            surprise={"orderly": 1.0, "disorderly": 1.5},
            turb_mult={"orderly": 1.0, "disorderly": 1.25}, turb_years=3,
            tau_window={"orderly": (2026, 2026), "disorderly": (2029, 2032)},
            credit_beta=0.20, chi=0.30, psi=0.50,       # credit loss; real-rate links
            nc_drag={"orderly": 0.0008, "disorderly": 0.0018},
            nc_amp={"orderly": 1.0, "disorderly": 1.3}, nc_equity=1.0,
            pv_disc=0.068, pv_growth=0.025, phys_damage_2055=0.0)
ALL_CHANNELS = {"repricing", "fossil", "credit", "turbulence", "nc_assets",
                "nc_liab", "rates_psi", "rates_chi", "physical"}


# ── NGFS data ────────────────────────────────────────────────────────────────
VARS = ["GDP|PPP|Counterfactual without damage", "Policy Cost|GDP Loss", "Price|Carbon",
        "Emissions|CO2", "Primary Energy|Coal", "Primary Energy|Oil", "Primary Energy|Gas"]

def download():
    import pyam
    df = pyam.read_iiasa("ngfs_phase_5", model=MODEL, scenario=list(SCENARIOS.values()),
                         variable=VARS, region="World").as_pandas()
    DATA.parent.mkdir(exist_ok=True); df.to_csv(DATA, index=False, encoding="utf-8")

def states():
    df = pd.read_csv(DATA, encoding="utf-8")
    df["scenario"] = df["scenario"].str.replace("?", "°", regex=False)
    get = lambda sc, v: np.interp(YEARS, *sc[sc.variable == v].sort_values("year")[["year", "value"]].values.T)
    out = {}
    for key, name in SCENARIOS.items():
        sc = df[df.scenario == name]
        gdp = get(sc, VARS[0])
        fossil = sum(get(sc, v) for v in VARS[4:])
        out[key] = dict(burden=get(sc, "Price|Carbon") * get(sc, "Emissions|CO2") / (1000 * gdp),
                        policy_loss=get(sc, "Policy Cost|GDP Loss") / gdp,
                        fossil_rel=fossil / fossil[0], gdp=gdp, carbon=get(sc, "Price|Carbon"),
                        coal=get(sc, VARS[4]), oil=get(sc, VARS[5]))
    return out

def pv_mean(x, i, c):
    """PV-weighted average of x from index i, held flat after the horizon."""
    q = (1 + c["pv_growth"]) / (1 + c["pv_disc"]); seg = x[i:]
    w = q ** np.arange(len(seg)); tail = w[-1] * q / (1 - q)
    return (w @ seg + tail * seg[-1]) / (w.sum() + tail)

def repricing(key, st, c, f):
    """One-off equity and credit loss for each possible recognition year."""
    out = {}
    for yr in range(c["tau_window"][key][0], c["tau_window"][key][1] + 1):
        i = yr - START
        broad = min(0.9, c["surprise"][key] * (c["theta_E"] * pv_mean(st["burden"], i, c)
                                               / c["profit_share"] + pv_mean(st["policy_loss"], i, c)))
        # Loss is measured against the 2025 volume path: before recognition the market
        # still prices fossil producers on unchanged volumes, so at tau it reprices both
        # the decline already realised and the decline still to come.
        fossil = min(0.95, max(0.0, 1 - pv_mean(st["fossil_rel"], i, c)))
        out[yr] = (i - 1, broad, fossil)
    return out


# ── Simulation ───────────────────────────────────────────────────────────────
def shocks(seed):
    r = np.random.default_rng(seed)
    return dict(ze=r.standard_normal((N, T)), zy=r.standard_normal((N, T)),
                zl=r.standard_normal((N, T)), zb=r.standard_normal((N, T)), u=r.random(N))

def simulate(z, key=None, st=None, c=COEF, f=FUND, m=MARKET, channels=None,
             tau_fixed=None, liability="pv"):
    """Funding-ratio paths (N x T+1). key=None gives the traditional ALM."""
    ch = set() if key is None else set(ALL_CHANNELS if channels is None else channels)
    burden = st["burden"][1:] if key else np.zeros(T)
    delta = c["nc_drag"][key] * c["nc_amp"][key] if key else 0.0
    dmg = np.diff(np.r_[0, c["phys_damage_2055"] * (np.arange(1, T + 1) / T) ** 2]) \
        if "physical" in ch else np.zeros(T)

    tau, j_eq, j_bd = np.full(N, -1), np.zeros(N), np.zeros(N)
    if ch & {"repricing", "fossil", "credit", "turbulence"}:
        rp = repricing(key, st, c, f); yrs = sorted(rp)
        k = (np.full(N, yrs.index(tau_fixed)) if tau_fixed in rp
             else np.minimum((z["u"] * len(yrs)).astype(int), len(yrs) - 1))
        tau = np.array([rp[yrs[i]][0] for i in k])
        broad = np.array([rp[yrs[i]][1] for i in k]); fossil = np.array([rp[yrs[i]][2] for i in k])
        j_eq = (1 - f["fossil_share"]) * broad * ("repricing" in ch) + f["fossil_share"] * fossil * ("fossil" in ch)
        j_bd = f["credit_share"] * c["credit_beta"] * j_eq * ("credit" in ch)

    ybar = m["y_bar"] + ("rates_chi" in ch) * c["chi"] * burden - ("rates_psi" in ch) * c["psi"] * delta
    y, A, L = np.full(N, m["y0"]), np.ones(N), np.ones(N)
    fr = np.ones((N, T + 1))
    for t in range(T):
        y_new = y + m["kappa"] * (ybar[t] - y) + m["sigma_y"] * z["zy"][:, t]
        dy, hit = y_new - y, (t == tau)
        turb = ("turbulence" in ch) & (t >= tau) & (t < tau + c["turb_years"]) & (tau >= 0)
        sig = m["sigma_e"] * np.where(turb, c["turb_mult"][key] if key else 1, 1)
        r_eq = y + m["erp"] + sig * z["ze"][:, t] - hit * j_eq - ("nc_assets" in ch) * c["nc_equity"] * delta - dmg[t]
        r_bd = y - f["D_B"] * dy - hit * j_bd + m["bond_noise"] * z["zb"][:, t]
        A *= 1 + f["w_eq"] * r_eq + f["w_bd"] * r_bd
        nc = ("nc_liab" in ch) * delta
        if liability == "pv":
            g = y + f["accrual_excess"] - f["D_L"] * dy - f["idx_wage"] * (nc + dmg[t])
        else:   # "target": reserve path growing at wage growth + target spread (GPIF-style)
            g = f["wage_growth"] + f["target_spread"] - nc - dmg[t]
        L *= 1 + g + f["liab_noise"] * z["zl"][:, t]
        y = y_new; fr[:, t + 1] = A / L
    return fr


# ── Metrics and experiments ──────────────────────────────────────────────────
pct = lambda x, p: float(np.percentile(x, p)) * 100
es = lambda x: float(x[x <= np.percentile(x, 5)].mean()) * 100

def summary(fr):
    f = fr[:, -1]
    return {"p5": pct(f, 5), "p50": pct(f, 50), "mean": f.mean() * 100,
            "p_below_80": (f < 0.8).mean() * 100, "climate_es": es(f),
            "es_by_horizon": {h: es(fr[:, h]) for h in (10, 20, 30)}}

def gap(st, key, n_seeds=20, liability="pv", **kw):
    """Climate Solvency Gap (p5 and median), mean and sd over seeds."""
    trad_kw = {k: v for k, v in kw.items() if k in ("f", "m")}
    g5, g50 = [], []
    for i in range(n_seeds):
        z = shocks(1042 + i)
        t, c = simulate(z, liability=liability, **trad_kw)[:, -1], simulate(z, key, st, liability=liability, **kw)[:, -1]
        g5.append(pct(t, 5) - pct(c, 5)); g50.append(pct(t, 50) - pct(c, 50))
    return {"p5": np.mean(g5), "p5_sd": np.std(g5, ddof=1), "p50": np.mean(g50)}

def decomposition(st, seed=42):
    z, on, rows = shocks(seed), set(), []
    prev = pct(simulate(z)[:, -1], 5)
    rows.append(("Traditional ALM", prev, 0.0))
    for ch in ["repricing", "fossil", "credit", "turbulence", "nc_assets", "nc_liab", "rates_psi", "rates_chi"]:
        on.add(ch); p5 = pct(simulate(z, "disorderly", st, channels=on)[:, -1], 5)
        rows.append(("+ " + ch, p5, p5 - prev)); prev = p5
    return rows

def tornado(st, seed=42):
    z = shocks(seed); trad = simulate(z)[:, -1]
    base = pct(trad, 5) - pct(simulate(z, "disorderly", st)[:, -1], 5)
    rows = {}
    for name in ["theta_E", "surprise", "turb_mult", "credit_beta", "chi", "psi",
                 "nc_drag", "nc_amp", "nc_equity", "idx_wage", "fossil_share"]:
        res = []
        for mult in (0.5, 1.5):
            c, f = copy.deepcopy(COEF), copy.deepcopy(FUND)
            if name in f: f[name] *= mult
            elif name in ("surprise", "turb_mult", "nc_amp"):       # perturb the excess over 1
                c[name] = {k: 1 + (v - 1) * mult for k, v in c[name].items()}
            elif isinstance(c[name], dict): c[name] = {k: v * mult for k, v in c[name].items()}
            else: c[name] *= mult
            t = trad          # these parameters do not enter the traditional ALM
            res.append(pct(t, 5) - pct(simulate(z, "disorderly", st, c=c, f=f)[:, -1], 5))
        rows[name] = res
    return base, rows

def severe():
    c = copy.deepcopy(COEF)
    c.update(theta_E=0.60, credit_beta=0.30,
             surprise={"orderly": 1.5, "disorderly": 2.5}, turb_mult={"orderly": 1.10, "disorderly": 1.50},
             nc_drag={k: v * 1.5 for k, v in COEF["nc_drag"].items()})
    return c


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st = states(); D = st["disorderly"]; res = {}
    z = shocks(42)
    res["table_seed42"] = {"traditional": summary(simulate(z)),
                           **{k: summary(simulate(z, k, st[k])) for k in SCENARIOS}}
    res["headline"] = {k: gap(st[k], k) for k in SCENARIOS}
    res["decomposition"] = decomposition(D)
    res["tornado_base"], res["tornado"] = tornado(D)
    sm = {**MARKET, "sigma_e": 0.072, "sigma_y": 0.004}
    res["sensitivities"] = {
        "smoothed vol 7.2%": gap(D, "disorderly", m=sm),
        "severe stress": gap(D, "disorderly", c=severe()),
        "no NCD": gap(D, "disorderly", c={**COEF, "nc_drag": {"orderly": 0, "disorderly": 0}}),
        **{f"psi = {p}": gap(D, "disorderly", c={**COEF, "psi": p}) for p in (0.0, 1.0, 1.5)},
        **{f"recognition {y}": gap(D, "disorderly", tau_fixed=y) for y in (2029, 2032)},
        **{f"physical -{int(d*100)}% GDP": gap(D, "disorderly", c={**COEF, "phys_damage_2055": d}) for d in (0.05, 0.10)}}
    gf = {**FUND, "w_eq": 0.5, "w_bd": 0.5, "D_B": 8.0, "wage_growth": 0.020, "target_spread": 0.019,
          "idx_wage": 1.0, "credit_share": 0.2}
    gm = {**MARKET, "y0": 0.0275, "y_bar": 0.0275, "erp": 0.035, "sigma_e": 0.18, "bond_noise": 0.045}
    res["gpif_style"] = {k: gap(st[k], k, n_seeds=10, f=gf, m=gm, liability="target") for k in SCENARIOS}

    from alternative_spec import run as alt
    res["alternative_permanent_drag"] = alt(st)

    h = res["headline"]
    t42 = res["table_seed42"]
    print(f"Seed 42 gap at p5: Disorderly {t42['traditional']['p5'] - t42['disorderly']['p5']:.2f} pp")
    print(f"Climate Solvency Gap at p5 (20 seeds): Disorderly {h['disorderly']['p5']:.1f} pp "
          f"(sd {h['disorderly']['p5_sd']:.2f}), Orderly {h['orderly']['p5']:.1f} pp; "
          f"at median {h['disorderly']['p50']:.1f} pp")
    for k, v in res["table_seed42"].items():
        print(f"  {k:12s} p5 {v['p5']:5.1f}  median {v['p50']:5.1f}  P(FR<80%) {v['p_below_80']:4.1f}  ES {v['climate_es']:4.1f}")
    print("Decomposition (p5, Disorderly, seed 42):")
    for name, p5, d in res["decomposition"]: print(f"  {name:16s} {p5:5.1f}  {d:+.1f}")
    print("Sensitivities (gap at p5):", {k: round(float(v["p5"]), 1) for k, v in res["sensitivities"].items()})
    print("Tornado range:", round(min(min(v) for v in res["tornado"].values()), 1), "to",
          round(max(max(v) for v in res["tornado"].values()), 1), "pp")
    print("GPIF-style gap at p5:", {k: round(float(v["p5"]), 1) for k, v in res["gpif_style"].items()})
    print("Permanent-drag specification:", {k: round(float(v), 1) for k, v in res["alternative_permanent_drag"].items()})
    Path("results.json").write_text(json.dumps(res, indent=1, default=float))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--download", action="store_true")
    if ap.parse_args().download: download()
    main()
