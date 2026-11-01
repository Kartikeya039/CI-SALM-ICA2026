"""Alternative specification used in Section 6.4 ("How much is calibration?").

Climate enters as a PERMANENT annual drift on returns, proportional to the
carbon price divided by each scenario's own maximum; equity volatility is the
smoothed 7.2% actuarial basis and liabilities grow as an undiscounted index.
It shows how strongly the size of the gap depends on calibration.
"""
import numpy as np

def _inputs(st, key):
    s = st[key]; gdp = s["gdp"]
    return dict(gdp_gap=np.diff(gdp) / gdp[:-1] - 0.024,
                policy=s["policy_loss"][1:], carbon=s["carbon_norm"][1:],
                fossil=s["fossil_decline"])

def _run(st, key, rng, eq_vol, bd_vol, beta_carbon):
    n, T = 10_000, 30
    if key is None:
        eq, bd, lg = np.full(T, 0.068), np.full(T, 0.038), np.full(T, 0.044)
        fossil, nc, pol, v, lvol = np.zeros(T), 0.0, None, 1.0, 0.02
    else:
        x = _inputs(st, key); s = {"orderly": 0.25, "disorderly": 0.32}[key]
        v = {"orderly": 1.01, "disorderly": 1.08}[key]
        nc = {"orderly": 0.0008, "disorderly": 0.0018}[key]
        eq = np.clip(0.068 + s * (0.25 * x["gdp_gap"] - 0.35 * x["policy"] - beta_carbon[0] * x["carbon"]), 0.01, 0.10)
        bd = np.clip(0.038 - s * (0.20 * x["policy"] + beta_carbon[1] * x["carbon"]), 0.005, 0.06)
        ls = {"orderly": 0.06, "disorderly": 0.18}[key]
        lg = np.clip(0.044 + ls * (0.08 * x["gdp_gap"] + 3.0 * nc), 0.038, 0.062)
        fossil, pol, lvol = 0.06 * x["fossil"], (x["policy"] if key == "disorderly" else None), 0.008 * v
        nc = nc * (1.3 if key == "disorderly" else 1.0)
    ev, bv = eq_vol * v, bd_vol * v
    cov = [[ev**2, 0.1 * ev * bv], [0.1 * ev * bv, bv**2]]
    A = np.ones(n)
    for t in range(T):
        zz = rng.multivariate_normal([0, 0], cov, size=n)
        A *= 1 + 0.6 * (eq[t] + zz[:, 0]) + 0.4 * (bd[t] + zz[:, 1]) - fossil[t]
    L = np.ones(n)
    for t in range(T):
        g = lg[t] - nc - (0.3 * pol[t] if pol is not None else 0.0)
        L *= 1 + g + rng.normal(0, lvol, n)
    return A / L

def gap(st, eq_vol=0.072, bd_vol=0.028, beta_carbon=(0.08, 0.03), seed=42):
    rng = np.random.default_rng(seed)
    trad = _run(st, None, rng, eq_vol, bd_vol, beta_carbon)
    _run(st, "orderly", rng, eq_vol, bd_vol, beta_carbon)          # keeps the draw order
    dis = _run(st, "disorderly", rng, eq_vol, bd_vol, beta_carbon)
    return (np.percentile(trad, 5) - np.percentile(dis, 5)) * 100

def run(st):
    for k in st:                                                 # extra inputs this spec needs
        s = st[k]
        s.setdefault("carbon_norm", s["carbon"] / max(s["carbon"].max(), 1))
        s.setdefault("fossil_decline", 0.5 * (-np.diff(s["coal"]) / np.maximum(s["coal"][:-1], 1)
                                             - np.diff(s["oil"]) / np.maximum(s["oil"][:-1], 1)))
    return {"smoothed vol (as specified)": gap(st),
            "market vol 16%": gap(st, eq_vol=0.16, bd_vol=0.05),
            "without carbon-drift term": np.mean([gap(st, beta_carbon=(0, 0), seed=s) for s in range(100, 120)])}
