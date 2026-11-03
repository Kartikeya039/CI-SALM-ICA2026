# CI-SALM: Climate-Integrated Stochastic ALM

Code for *"A Climate-Integrated Stochastic ALM Approach to Bridge the Pension Climate Solvency Gap"* (Kartikeya Singh, 33rd International Congress of Actuaries, Tokyo 2026).

CI-SALM maps NGFS Phase V transition scenarios into a stochastic asset-liability model of a stylised defined-benefit pension fund. It compares the result with a stationary (traditional) ALM run on the same random shocks.

## Run

```
pip install -r requirements.txt
python cisalm.py
```

This reproduces every number in the paper and slides in about 15 seconds and saves them to `results.json`. To refresh the NGFS data from IIASA first: `pip install pyam-iamc`, then `python cisalm.py --download`.

## Main results (Disorderly = NGFS Delayed Transition)

| Metric | Traditional | Orderly | Disorderly |
|---|---|---|---|
| 5th-percentile funding ratio, 2055 (seed 42) | 62.2% | 60.7% | 58.2% |
| Median funding ratio, 2055 (seed 42) | 150.0% | 146.3% | 142.8% |
| Climate Solvency Gap at p5, seed 42 | reference | 1.5 pp | 4.0 pp |
| Climate Solvency Gap at p5, mean of 20 seeds (1042 to 1061) | reference | 1.6 pp | 4.5 pp (sd 0.25) |

The first two rows, the decomposition and the tornado use seed 42, so they centre on a 4.0 pp gap. The headline and the sensitivities are averages over 20 other seeds, so they centre on 4.5 pp. The difference is ordinary sampling variation in a 5th-percentile estimate.

Severe-stress calibration: 9.9 pp. Every ±50% change to a judgment parameter keeps the gap between 2.9 and 5.2 pp.

## How the model works

- **Assets:** a 60/40 portfolio with market volatility (16%) and a stochastic real rate. When the transition is recognised, assets take a one-off transition shock, sized from NGFS carbon cost (price × emissions ÷ GDP) and policy GDP loss. A 6% fossil sleeve loses the present value of its stranded coal, oil and gas volumes. Recognition comes in 2026 (Orderly), or at a random year from 2029 to 2032 with a surprise factor and turbulence (Disorderly).
- **Liabilities:** a present value with duration 14. A Natural Capital Dependency (NCD) drag slows productivity, which lowers wage-linked benefits, dividend growth and the real rate.
- **Comparison:** the traditional ALM uses identical shocks with every climate term switched off.

## Files

| File | Contents |
|---|---|
| `cisalm.py` | Model, experiments (headline, decomposition, tornado, sensitivities, GPIF-style variant) |
| `alternative_spec.py` | Alternative specification with a permanent carbon-price drag (paper Section 6.4: 23.8 pp), showing how strongly the gap's size depends on calibration |
| `data/ngfs_phase5.csv` | NGFS Phase V, MESSAGEix-GLOBIOM 2.0-M-R12-NGFS, World: Below 2°C and Delayed Transition |

## Data and caveats

NGFS Climate Scenarios, Phase V (November 2024), via the IIASA NGFS Scenario Explorer. Please cite NGFS and IIASA when reusing the data.

This is an illustrative research model, not a forecast, and not investment or actuarial advice. Physical risk is excluded by design. Judgment parameters are calibrated to orders of magnitude in the literature, and all are listed at the top of `cisalm.py`.
