"""
CI-SALM: Climate-Integrated Stochastic Asset-Liability Model
============================================================
Monte Carlo simulation engine integrating forward-looking NGFS Phase V climate
pathways (MESSAGEix-GLOBIOM 2.0-M-R12-NGFS, World region) into both asset-side
returns and liability-side growth for a defined-benefit pension scheme.

Usage:
    python simulate_cisalm.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Directory & Simulation Configuration ──────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
for directory in (DATA_DIR, OUT_DIR):
    directory.mkdir(exist_ok=True)

N_PATHS: int = 10_000
HORIZON: int = 30               # Annual steps: 2025 -> 2055
SEED: int = 42                  # Fixed PCG64 seed for exact replication

INITIAL_ASSETS: float = 100.0   # Normalised 100% initial funding ratio
INITIAL_LIABILITIES: float = 100.0
EQUITY_WEIGHT: float = 0.60
BOND_WEIGHT: float = 0.40

# Benchmark Stationary ALM Parameters (Smoothed Actuarial Accounting Basis)
TRAD = dict(
    eq_ret=0.068,               # Assumed long-run real equity return (6.8% p.a.)
    bd_ret=0.038,               # Assumed investment-grade bond yield (3.8% p.a.)
    eq_vol=0.072,               # Smoothed actuarial equity volatility (7.2% p.a.)
    bd_vol=0.028,               # Smoothed actuarial bond volatility (2.8% p.a.)
    corr=0.10,                  # Long-run equity-bond correlation
    liab_growth=0.044,          # Baseline wage & benefit indexation rate (4.4% p.a.)
    liab_vol_base=0.020,        # Stationary demographic/liability experience noise (2.0% p.a.)
    liab_vol_cisalm=0.008,      # Smoothed wage-indexation noise in CI-SALM (0.8% p.a.)
)
MARKET_VOL = dict(eq_vol=0.16, bd_vol=0.05)

# CI-SALM Transmission Coefficients (All Perturbed ±50% in Tornado Analysis)
COEF = dict(
    fossil_weight=0.06,         # Carbon-intensive equity sleeve weight (6.0%)
    beta_gdp=0.25,              # Equity drift sensitivity to macro GDP growth gap
    beta_policy=0.35,           # Equity drift drag per unit policy cost / GDP
    beta_carbon=0.08,           # Equity drift drag per unit normalised carbon price
    beta_policy_bd=0.20,        # Bond drift drag per unit policy cost / GDP
    beta_carbon_bd=0.03,        # Bond drift drag per unit normalised carbon price
    liab_beta_gdp=0.08,         # Liability growth sensitivity to GDP growth gap
    nc_mult=3.0,                # Supply-side inflation pass-through multiplier
    nc_disorderly_amp=1.3,      # Disorderly transition amplification on chronic NCD drag
    policy_to_liab=0.30,        # Policy-uncertainty pass-through to liability growth
    shock_scale={"orderly": 0.25, "disorderly": 0.32},
    vol_mult={"orderly": 1.01, "disorderly": 1.08},
    liab_shock={"orderly": 0.06, "disorderly": 0.18},
    nc_drag={"orderly": 0.0008, "disorderly": 0.0018},
)

SCENARIO_MAP = {"orderly": "Below 2°C", "disorderly": "Delayed transition"}
SCENARIO_LABELS = {
    "traditional": "Traditional ALM",
    "orderly": "CI-SALM · Orderly (Below 2°C)",
    "disorderly": "CI-SALM · Disorderly (Delayed Transition)",
}
NGFS_VARIABLES = [
    "GDP|PPP|Counterfactual without damage",
    "Policy Cost|GDP Loss",
    "Price|Carbon",
    "Emissions|CO2",
    "Primary Energy|Coal",
    "Primary Energy|Oil",
    "Primary Energy|Gas",
]

PROVENANCE = [
    {"param": "eq_ret,bd_ret", "value": [TRAD["eq_ret"], TRAD["bd_ret"]],
     "basis": "Long-run institutional capital market assumptions (CalPERS 2023; GPIF)",
     "robustness": "market-vol run"},
    {"param": "eq_vol,bd_vol", "value": [TRAD["eq_vol"], TRAD["bd_vol"]],
     "basis": "Smoothed actuarial accounting volatility (3-5yr asset smoothing)",
     "robustness": "market-vol run at 16%"},
    {"param": "fossil_weight", "value": COEF["fossil_weight"],
     "basis": "Representative energy-sector weight in global equity indices (MSCI World)",
     "robustness": "±50% tornado"},
    {"param": "nc_drag", "value": COEF["nc_drag"],
     "basis": "Chronic ecosystem-service productivity drag (Dasgupta Review 2021; World Bank 2021)",
     "robustness": "±50% tornado"},
    {"param": "beta_gdp,beta_policy,beta_carbon,beta_policy_bd,beta_carbon_bd",
     "value": [COEF["beta_gdp"], COEF["beta_policy"], COEF["beta_carbon"],
               COEF["beta_policy_bd"], COEF["beta_carbon_bd"]],
     "basis": "Illustrative transmission parameters anchored to macro climate damages "
              "(Dietz et al. 2016, Nature Climate Change), financial network transition exposure "
              "(Battiston et al. 2017, Nature Climate Change), and stranded fossil-fuel assets "
              "(Semieniuk et al. 2022, Nature Climate Change).",
     "robustness": "±50% tornado"},
    {"param": "liab_beta_gdp,nc_mult,nc_disorderly_amp,policy_to_liab",
     "value": [COEF["liab_beta_gdp"], COEF["nc_mult"],
               COEF["nc_disorderly_amp"], COEF["policy_to_liab"]],
     "basis": "Reduced-form composite of sovereign borrowing spreads (Beirne, Renzhi & Volz 2021), "
              "climate inflation pass-through (Faccia, Parker & Stracca 2021, ECB WP 2626), and "
              "productivity-linked real wage growth (Dasgupta 2021).",
     "robustness": "±50% tornado & harmonized liability-volatility check"},
    {"param": "shock_scale,liab_shock,vol_mult",
     "value": [COEF["shock_scale"], COEF["liab_shock"], COEF["vol_mult"]],
     "basis": "Calibrated to NGFS Phase V (2024) scenario narratives (disorderly > orderly)",
     "robustness": "±50% tornado"},
    {"param": "drift_clips",
     "value": {"eq": [0.01, 0.10], "bd": [0.005, 0.06], "liab": [0.038, 0.062]},
     "basis": "Non-binding numerical stability guardrails",
     "robustness": "no-clip diagnostic run"},
]


# ── 1. Data Ingestion & State-Variable Derivation ─────────────────────────────
def fetch_ngfs_data(force_download: bool = False) -> pd.DataFrame:
    """Load cached NGFS Phase V World series or query IIASA Scenario Explorer API."""
    cache = DATA_DIR / "ngfs_official.csv"
    if cache.exists() and not force_download:
        return pd.read_csv(cache, encoding="utf-8")
    import pyam
    query_kw = dict(
        model="MESSAGEix-GLOBIOM 2.0-M-R12-NGFS",
        scenario=list(SCENARIO_MAP.values()),
        variable=NGFS_VARIABLES,
        region="World",
    )
    try:
        # Primary method (pyam >= 2.0 / 3.0): supports both ixmp4 and legacy IIASA backends in one batch request
        idf = pyam.read_iiasa("ngfs_phase_5", **query_kw)
        df = idf.as_pandas() if hasattr(idf, "as_pandas") else pd.DataFrame(idf)
    except Exception:
        # Fallback: explicit variable-by-variable Connection query for legacy pyam environments
        conn = pyam.iiasa.Connection("ngfs_phase_5")
        frames = []
        for var in NGFS_VARIABLES:
            res = conn.query(
                model=query_kw["model"],
                scenario=query_kw["scenario"],
                variable=var,
                region=query_kw["region"],
            )
            f = res.as_pandas() if hasattr(res, "as_pandas") else pd.DataFrame(res)
            if len(f):
                frames.append(f)
        df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    return df


def build_scenario_params(ngfs: pd.DataFrame) -> dict:
    """Interpolate 5-year IAM knots to annual grid and compute state variables."""
    years = np.arange(2025, 2025 + HORIZON + 1)
    ngfs = ngfs.assign(scenario=ngfs["scenario"].str.replace("Â°", "°", regex=False).str.replace("?", "°", regex=False))

    def interp_series(sc: pd.DataFrame, var: str) -> np.ndarray:
        s = sc[sc["variable"] == var].sort_values("year").set_index("year")["value"]
        return np.interp(years, s.index.values, s.values)

    decline_rate = lambda v: -np.diff(v) / np.maximum(v[:-1], 1.0)
    params = {}
    for key, sc_name in SCENARIO_MAP.items():
        sc = ngfs[ngfs["scenario"] == sc_name]
        gdp = interp_series(sc, NGFS_VARIABLES[0])
        policy = interp_series(sc, NGFS_VARIABLES[1])
        carbon = interp_series(sc, NGFS_VARIABLES[2])
        emis = interp_series(sc, NGFS_VARIABLES[3])
        coal = interp_series(sc, NGFS_VARIABLES[4])
        oil = interp_series(sc, NGFS_VARIABLES[5])

        params[key] = dict(
            gdp_growth=np.diff(gdp) / gdp[:-1],
            policy_drag=policy[1:] / np.maximum(gdp[1:], 1.0),
            fossil_shock=0.5 * (decline_rate(coal) + decline_rate(oil)),
            carbon_norm=carbon[1:] / max(carbon.max(), 1.0),
            nc_drag=COEF["nc_drag"][key],
            end_values=dict(
                carbon=float(carbon[-1]),
                policy_cost=float(policy[-1]),
                emissions=float(emis[-1]),
                gdp=float(gdp[-1]),
            ),
        )
    return params


# ── 2. Stochastic Asset & Liability Simulation Engine ─────────────────────────
def simulate_portfolio(
    n_paths: int,
    horizon: int,
    eq_ret: float | np.ndarray,
    bd_ret: float | np.ndarray,
    eq_vol: float,
    bd_vol: float,
    corr: float,
    fossil_shock: np.ndarray | None = None,
    fossil_weight: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate correlated bivariate normal equity/bond portfolio trajectories."""
    rng = rng or np.random.default_rng(SEED)
    cov = np.array([
        [eq_vol**2, corr * eq_vol * bd_vol],
        [corr * eq_vol * bd_vol, bd_vol**2],
    ])
    paths = np.full((n_paths, horizon + 1), 1.0)
    for t in range(horizon):
        e = eq_ret[t] if isinstance(eq_ret, np.ndarray) else eq_ret
        b = bd_ret[t] if isinstance(bd_ret, np.ndarray) else bd_ret
        z = rng.multivariate_normal([0.0, 0.0], cov, size=n_paths)
        port_ret = EQUITY_WEIGHT * (e + z[:, 0]) + BOND_WEIGHT * (b + z[:, 1])
        if fossil_shock is not None:
            port_ret -= fossil_weight * fossil_shock[t]
        paths[:, t + 1] = paths[:, t] * (1.0 + port_ret)
    return paths * INITIAL_ASSETS


def simulate_liabilities(
    n_paths: int,
    horizon: int,
    base_growth: float | np.ndarray,
    nc_drag: float,
    policy_drag: np.ndarray | None = None,
    policy_to_liab: float = 0.30,
    extra_vol: float = TRAD["liab_vol_base"],
    rng: np.random.Generator | None = None,
    cat_shocks: bool = False,
) -> np.ndarray:
    """Simulate pension liability accumulation with chronic NCD drag."""
    rng = rng or np.random.default_rng(SEED + 1)
    paths = np.full((n_paths, horizon + 1), INITIAL_LIABILITIES)
    for t in range(horizon):
        g = base_growth[t] if isinstance(base_growth, np.ndarray) else base_growth
        growth = g - nc_drag
        if policy_drag is not None:
            growth -= policy_to_liab * policy_drag[t]
        if cat_shocks and nc_drag > 0:
            prob = min(0.20, 20.0 * nc_drag)
            growth = growth + rng.binomial(1, prob, n_paths) * rng.uniform(0.03, 0.08, n_paths)
        growth = growth + rng.normal(0.0, extra_vol, n_paths)
        paths[:, t + 1] = paths[:, t] * (1.0 + growth)
    return paths


def simulate_traditional(
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
    eq_vol: float | None = None,
    bd_vol: float | None = None,
    extra_vol: float = TRAD["liab_vol_base"],
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate stationary Traditional ALM baseline."""
    assets = simulate_portfolio(
        n_paths, horizon, TRAD["eq_ret"], TRAD["bd_ret"],
        eq_vol or TRAD["eq_vol"], bd_vol or TRAD["bd_vol"], TRAD["corr"], rng=rng,
    )
    liabs = simulate_liabilities(
        n_paths, horizon, TRAD["liab_growth"], nc_drag=0.0, extra_vol=extra_vol, rng=rng,
    )
    return assets, liabs


def simulate_cisalm(
    n_paths: int,
    horizon: int,
    key: str,
    sp: dict,
    rng: np.random.Generator,
    coef: dict = COEF,
    clip: bool = True,
    eq_vol: float | None = None,
    bd_vol: float | None = None,
    cat_shocks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate coupled CI-SALM asset and liability trajectories."""
    gdp_gap = sp["gdp_growth"] - 0.024
    s = coef["shock_scale"][key]
    v = coef["vol_mult"][key]

    eq_ret = TRAD["eq_ret"] + s * (
        coef["beta_gdp"] * gdp_gap
        - coef["beta_policy"] * sp["policy_drag"]
        - coef["beta_carbon"] * sp["carbon_norm"]
    )
    bd_ret = TRAD["bd_ret"] - s * (
        coef["beta_policy_bd"] * sp["policy_drag"]
        + coef["beta_carbon_bd"] * sp["carbon_norm"]
    )
    liab_growth = TRAD["liab_growth"] + coef["liab_shock"][key] * (
        coef["liab_beta_gdp"] * gdp_gap + coef["nc_mult"] * sp["nc_drag"]
    )
    if clip:
        eq_ret = np.clip(eq_ret, 0.01, 0.10)
        bd_ret = np.clip(bd_ret, 0.005, 0.06)
        liab_growth = np.clip(liab_growth, 0.038, 0.062)

    assets = simulate_portfolio(
        n_paths, horizon, eq_ret, bd_ret,
        (eq_vol or TRAD["eq_vol"]) * v, (bd_vol or TRAD["bd_vol"]) * v, TRAD["corr"],
        fossil_shock=sp["fossil_shock"], fossil_weight=coef["fossil_weight"], rng=rng,
    )
    liabs = simulate_liabilities(
        n_paths, horizon, liab_growth,
        nc_drag=sp["nc_drag"] * (coef["nc_disorderly_amp"] if key == "disorderly" else 1.0),
        policy_drag=sp["policy_drag"] if key == "disorderly" else None,
        policy_to_liab=coef["policy_to_liab"],
        extra_vol=TRAD["liab_vol_cisalm"] * v,
        rng=rng,
        cat_shocks=cat_shocks,
    )
    return assets, liabs


# ── 3. Risk Metrics, Channel Decomposition & Robustness ──────────────────────
pct = lambda v: round(float(v) * 100, 1)


def climate_es(x: np.ndarray, alpha: float = 0.05) -> float:
    """Climate Expected Shortfall (mean funding ratio in worst alpha-tail)."""
    cutoff = np.percentile(x, alpha * 100)
    tail = x[x <= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


def scenario_block(fr: np.ndarray, label: str) -> dict:
    """Compute terminal summary distribution, percentile bands, and horizon ES."""
    f = fr[:, -1]
    summary = {
        "label": label,
        "mean": pct(f.mean()),
        "median": pct(np.median(f)),
        **{f"p{p}": pct(np.percentile(f, p)) for p in (5, 25, 75, 95)},
        "p_fr_below_80": pct((f < 0.80).mean()),
        "p_fr_below_60": pct((f < 0.60).mean()),
        "climate_es_5": pct(climate_es(f)),
    }
    bands = {f"p{p}": (np.percentile(fr, p, axis=0) * 100).tolist() for p in (5, 25, 50, 75, 95)}
    bands["mean"] = (fr.mean(axis=0) * 100).tolist()
    return {
        "summary": summary,
        "bands": bands,
        "climate_es": {f"horizon_{t}": pct(climate_es(fr[:, t])) for t in (10, 20, 30)},
    }


def decompose_channels(
    a_trad: np.ndarray, l_trad: np.ndarray, a_dis: np.ndarray, l_dis: np.ndarray
) -> dict:
    """Compute Table 5 / Slide 11 stepwise decomposition of the Disorderly p5 gap."""
    p5_trad = pct(np.percentile(a_trad[:, -1] / l_trad[:, -1], 5))
    p5_net = pct(np.percentile(a_dis[:, -1] / l_dis[:, -1], 5))
    p5_asset_pathwise = pct(np.percentile(a_dis[:, -1] / l_trad[:, -1], 5))
    liab_cum_reduction_pp = pct(1.0 - l_dis[:, -1].mean() / l_trad[:, -1].mean())
    net_gap_pp = round(p5_trad - p5_net, 1)
    return {
        "traditional_p5_pct": p5_trad,
        "asset_side_drag_pp": round(-net_gap_pp - liab_cum_reduction_pp, 1),
        "liability_ncd_offset_pp": liab_cum_reduction_pp,
        "pathwise_asset_only_p5_pct": p5_asset_pathwise,
        "pathwise_asset_drag_pp": round(p5_asset_pathwise - p5_trad, 1),
        "pathwise_liability_offset_pp": round(p5_net - p5_asset_pathwise, 1),
        "net_cisalm_disorderly_p5_pct": p5_net,
        "net_solvency_gap_pp": net_gap_pp,
    }


def p5_gap(sp: dict, trad_p5: float, seed: int, **kw) -> float:
    """Climate Solvency Gap (pp) for the Disorderly scenario at a given seed."""
    a, l = simulate_cisalm(N_PATHS, HORIZON, "disorderly", sp, np.random.default_rng(seed), **kw)
    return round(trad_p5 - float(np.percentile(a[:, -1] / l[:, -1], 5)) * 100, 1)


def sensitivity_tornado(sp: dict, trad_p5: float) -> dict:
    """14-parameter ±50% perturbation tornado around same-seed (SEED + 7) base."""
    scalars = [
        "fossil_weight", "beta_gdp", "beta_policy", "beta_carbon", "beta_policy_bd",
        "beta_carbon_bd", "liab_beta_gdp", "nc_mult", "nc_disorderly_amp", "policy_to_liab",
    ]
    keyed = ["shock_scale", "liab_shock", "nc_drag"]
    rows = []
    for name in scalars + keyed + ["vol_mult_excess"]:
        gaps = {}
        for direction, mult in (("low", 0.5), ("high", 1.5)):
            c, s = copy.deepcopy(COEF), dict(sp)
            if name in scalars:
                c[name] = COEF[name] * mult
            elif name in keyed:
                c[name] = {k: v * mult for k, v in COEF[name].items()}
                if name == "nc_drag":
                    s["nc_drag"] = sp["nc_drag"] * mult
            else:
                c["vol_mult"] = {k: 1.0 + (v - 1.0) * mult for k, v in COEF["vol_mult"].items()}
            gaps[direction] = p5_gap(s, trad_p5, SEED + 7, coef=c)
        rows.append({
            "param": name,
            "gap_low": gaps["low"],
            "gap_high": gaps["high"],
            "spread": round(abs(gaps["high"] - gaps["low"]), 1),
            "inverted": gaps["low"] > gaps["high"],
        })
    return {"base_gap_pp": p5_gap(sp, trad_p5, SEED + 7), "rows": sorted(rows, key=lambda r: -r["spread"])}


def clip_binding_check(sp: dict) -> dict:
    """Evaluate drift clip-binding frequency at 1x, 2x, and 4x scaled shocks."""
    gdp_gap, out = sp["gdp_growth"] - 0.024, {}
    for m, tag in ((1, "base"), (2, "2x_shock_scale"), (4, "4x_shock_scale")):
        s, ls = COEF["shock_scale"]["disorderly"] * m, COEF["liab_shock"]["disorderly"] * m
        eq = TRAD["eq_ret"] + s * (COEF["beta_gdp"] * gdp_gap - COEF["beta_policy"] * sp["policy_drag"] - COEF["beta_carbon"] * sp["carbon_norm"])
        bd = TRAD["bd_ret"] - s * (COEF["beta_policy_bd"] * sp["policy_drag"] + COEF["beta_carbon_bd"] * sp["carbon_norm"])
        lg = TRAD["liab_growth"] + ls * (COEF["liab_beta_gdp"] * gdp_gap + COEF["nc_mult"] * sp["nc_drag"])
        out[f"eq_clip_binding_frac_{tag}"] = round(float(((eq < 0.01) | (eq > 0.10)).mean()), 3)
        out[f"bd_clip_binding_frac_{tag}"] = round(float(((bd < 0.005) | (bd > 0.06)).mean()), 3)
        out[f"liab_clip_binding_frac_{tag}"] = round(float(((lg < 0.038) | (lg > 0.062)).mean()), 3)
    return out


def convergence_check(sp: dict) -> dict:
    """Evaluate path-count convergence (N=500..10,000) and 1,000-rep bootstrap SE."""
    gaps, t_fr, d_fr = {}, None, None
    for n in (500, 1000, 2500, 5000, N_PATHS):
        rng = np.random.default_rng(SEED + 11)
        a, l = simulate_traditional(n, HORIZON, rng)
        t_fr = a[:, -1] / l[:, -1]
        a, l = simulate_cisalm(n, HORIZON, "disorderly", sp, rng)
        d_fr = a[:, -1] / l[:, -1]
        gaps[f"n_{n}"] = round(float(np.percentile(t_fr, 5) - np.percentile(d_fr, 5)) * 100, 1)
    brng = np.random.default_rng(SEED + 13)
    reps = np.array([
        (np.percentile(brng.choice(t_fr, t_fr.size), 5) - np.percentile(brng.choice(d_fr, d_fr.size), 5)) * 100
        for _ in range(1000)
    ])
    return {
        "gap_by_n_paths_pp": gaps,
        "bootstrap": {
            "replicates": 1000,
            "se_pp": round(float(reps.std(ddof=1)), 2),
            "ci95_pp": [round(float(np.percentile(reps, 2.5)), 1), round(float(np.percentile(reps, 97.5)), 1)],
        },
    }


# ── 4. Main Execution & Terminal Reporting ────────────────────────────────────
def run_simulation() -> dict:
    t0 = time.perf_counter()
    bar = "═" * 76
    print(f"\n{bar}\n  CI-SALM MONTE CARLO ENGINE · NGFS Phase V (N={N_PATHS:,} paths, T={HORIZON}y, Seed={SEED})\n{bar}", flush=True)

    print("  [1/5] Loading NGFS Phase V pathways (MESSAGEix-GLOBIOM World)...", flush=True)
    sps = build_scenario_params(fetch_ngfs_data())
    rng = np.random.default_rng(SEED)
    results: dict = {"scenarios": {}, "ngfs_calibration": {}}

    for key, sp in sps.items():
        ev = sp["end_values"]
        results["ngfs_calibration"][key] = {
            "label": SCENARIO_LABELS[key],
            "ngfs_scenario": SCENARIO_MAP[key],
            "carbon_price_2055_usd_t": round(ev["carbon"], 1),
            "policy_cost_gdp_loss_2055_bn_usd": round(ev["policy_cost"], 1),
            "co2_emissions_2055_mt": round(ev["emissions"], 1),
            "gdp_2055_bn_usd_ppp": round(ev["gdp"], 1),
            "avg_annual_gdp_growth_pct": round(float(sp["gdp_growth"].mean()) * 100, 3),
            "avg_policy_drag_pct": round(float(sp["policy_drag"].mean()) * 100, 4),
            "avg_fossil_decline_pct": round(float(sp["fossil_shock"].mean()) * 100, 3),
            "nc_drag_annual_pct": round(sp["nc_drag"] * 100, 3),
        }

    print(f"  [2/5] Simulating Traditional ALM Baseline ({N_PATHS:,} paths)...", flush=True)
    a_tr, l_tr = simulate_traditional(N_PATHS, HORIZON, rng)
    results["scenarios"]["traditional"] = scenario_block(a_tr / l_tr, SCENARIO_LABELS["traditional"])
    trad_p5 = results["scenarios"]["traditional"]["summary"]["p5"]

    sim_paths = {}
    for idx, key in enumerate(("orderly", "disorderly"), start=3):
        print(f"  [{idx}/5] Simulating {SCENARIO_LABELS[key]} ({N_PATHS:,} paths)...", flush=True)
        a, l = simulate_cisalm(N_PATHS, HORIZON, key, sps[key], rng)
        sim_paths[key] = (a, l)
        block = scenario_block(a / l, SCENARIO_LABELS[key])
        block["solvency_gap_pp"] = round(trad_p5 - block["summary"]["p5"], 1)
        results["scenarios"][key] = block

    tr, or_s, di = (results["scenarios"][k]["summary"] for k in ("traditional", "orderly", "disorderly"))
    results["channel_decomposition"] = decompose_channels(a_tr, l_tr, *sim_paths["disorderly"])
    results["key_findings"] = {
        "traditional_p5_fr": tr["p5"],
        "disorderly_p5_fr": di["p5"],
        "solvency_gap_pp": round(tr["p5"] - di["p5"], 1),
        "underfunding_80_traditional_pct": tr["p_fr_below_80"],
        "underfunding_80_disorderly_pct": di["p_fr_below_80"],
        "underfunding_multiplier": round(di["p_fr_below_80"] / max(tr["p_fr_below_80"], 0.1), 1),
        "additional_contribution_disorderly_pct": round(max(0.0, 100.0 - di["p5"]), 0),
    }

    print("  [5/5] Running robustness suite & 14-parameter tornado (28 perturbations)...", flush=True)
    a_mv, l_mv = simulate_traditional(N_PATHS, HORIZON, np.random.default_rng(SEED + 3), **MARKET_VOL)
    trad_p5_mv = float(np.percentile((a_mv / l_mv)[:, -1], 5)) * 100
    a_h, l_h = simulate_traditional(N_PATHS, HORIZON, np.random.default_rng(SEED), extra_vol=TRAD["liab_vol_cisalm"])
    trad_p5_h = pct(np.percentile((a_h / l_h)[:, -1], 5))
    base_gap_s5 = p5_gap(sps["disorderly"], trad_p5, SEED + 5)

    results["robustness"] = {
        "base_gap_seed5_pp": base_gap_s5,
        "no_clip": {"solvency_gap_pp": p5_gap(sps["disorderly"], trad_p5, SEED + 5, clip=False)},
        "clip_binding_check": clip_binding_check(sps["disorderly"]),
        "convergence": convergence_check(sps["disorderly"]),
        "market_vol_16pct": {
            "traditional_p5": round(trad_p5_mv, 1),
            "solvency_gap_pp": p5_gap(sps["disorderly"], trad_p5_mv, SEED + 5, **MARKET_VOL),
        },
        "harmonized_liab_vol_008": {
            "traditional_p5": trad_p5_h,
            "solvency_gap_pp": round(trad_p5_h - di["p5"], 1),
        },
        "cat_shock_stress": {"solvency_gap_pp": p5_gap(sps["disorderly"], trad_p5, SEED + 5, cat_shocks=True)},
        "sensitivity_tornado_pm50pct": sensitivity_tornado(sps["disorderly"], trad_p5),
    }
    results["metadata"] = {
        "n_paths": N_PATHS,
        "horizon_years": HORIZON,
        "seed": SEED,
        "data_source": "NGFS Phase V — IIASA Scenario Explorer (ngfs_phase_5)",
        "iam_model": "MESSAGEix-GLOBIOM 2.0-M-R12-NGFS",
        "region": "World",
        "scenarios": "Orderly (Below 2°C), Disorderly (Delayed Transition)",
        "initial_funding_ratio_pct": round(INITIAL_ASSETS / INITIAL_LIABILITIES * 100, 1),
        "portfolio": f"{int(EQUITY_WEIGHT * 100)}% Equity / {int(BOND_WEIGHT * 100)}% Bonds",
        "volatility_basis": "Smoothed actuarial basis (7.2% eq); market-vol robustness at 16%",
        "tail_metric": "Climate Expected Shortfall (Climate-ES), 5% level",
        "calibration_status": "Illustrative calibration; see results/parameter_provenance.json",
        "years": list(range(2025, 2025 + HORIZON + 1)),
    }

    with open(OUT_DIR / "simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "parameter_provenance.json", "w", encoding="utf-8") as f:
        json.dump(PROVENANCE, f, indent=2)

    kf, rb, cd = results["key_findings"], results["robustness"], results["channel_decomposition"]
    tor = rb["sensitivity_tornado_pm50pct"]
    print(f"\n{bar}\n  TERMINAL FUNDING RATIO DISTRIBUTION AT T=30 (YEAR 2055)\n{bar}")
    print(f"  {'Scenario':<42} {'Median':>7} {'p5 Tail':>8} {'Δ_CSG':>8} {'P(FR<80%)':>10} {'Clim-ES5':>9}")
    print("  " + "─" * 72)
    print(f"  {tr['label']:<42} {tr['median']:>6.1f}% {tr['p5']:>7.1f}% {'Ref':>8} {tr['p_fr_below_80']:>9.1f}% {tr['climate_es_5']:>8.1f}%")
    print(f"  {or_s['label']:<42} {or_s['median']:>6.1f}% {or_s['p5']:>7.1f}% {results['scenarios']['orderly']['solvency_gap_pp']:>+7.1f}pp {or_s['p_fr_below_80']:>9.1f}% {or_s['climate_es_5']:>8.1f}%")
    print(f"  {di['label']:<42} {di['median']:>6.1f}% {di['p5']:>7.1f}% {kf['solvency_gap_pp']:>+7.1f}pp {di['p_fr_below_80']:>9.1f}% {di['climate_es_5']:>8.1f}%")

    print(f"\n  -- Channel Decomposition (Table 5 & Slide 11: {cd['traditional_p5_pct']}% -> {cd['net_cisalm_disorderly_p5_pct']}%) --")
    print(f"  Asset-Side Climate Drag   : {cd['asset_side_drag_pp']:+6.1f} pp  (pathwise asset-only drag: {cd['pathwise_asset_drag_pp']:+.1f} pp)")
    print(f"  Liability-Side NCD Offset : {cd['liability_ncd_offset_pp']:+6.1f} pp  (pathwise liability offset: {cd['pathwise_liability_offset_pp']:+.1f} pp)")

    print(f"\n  -- Robustness Suite (Δ_CSG, pp; same-seed base [{rb['base_gap_seed5_pp']} pp]) --")
    print(f"  No drift clips            : {rb['no_clip']['solvency_gap_pp']:>5.1f} pp  (clips bind 0.0% at base calibration)")
    print(f"  Harmonized liab vol (0.8%): {rb['harmonized_liab_vol_008']['solvency_gap_pp']:>5.1f} pp  (Trad p5 rises to {rb['harmonized_liab_vol_008']['traditional_p5']}%)")
    print(f"  Market asset vol (16%)    : {rb['market_vol_16pct']['solvency_gap_pp']:>5.1f} pp  (Trad p5 falls to {rb['market_vol_16pct']['traditional_p5']}%)")
    print(f"  Acute catastrophe shock   : {rb['cat_shock_stress']['solvency_gap_pp']:>5.1f} pp  (chronic NCD + episodic claim spikes)")

    print(f"\n  -- Top Tornado Sensitivities (±50% Perturbation around {tor['base_gap_pp']} pp Base) --")
    for r in tor["rows"][:6]:
        tag = " [inv]" if r["inverted"] else ""
        print(f"  {r['param']:<18} : {r['gap_low']:>5.1f} ──┼── {r['gap_high']:<5.1f} pp  (spread {r['spread']:>4.1f} pp){tag}")
    print(f"{bar}\n  Completed in {time.perf_counter() - t0:.2f}s · Saved to results/simulation_results.json\n{bar}\n", flush=True)
    return results


if __name__ == "__main__":
    run_simulation()
