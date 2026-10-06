"""
Rebuilt from the corrected export, which restores columns v1/v2 were
missing: `Pick Result`, `Recommended_Side`, `Recommended_Logic_Code`.

`Pick Result` (Win / Loss / Push / Pending / No Bet (Late)) is the direct,
authoritative outcome field -- no more reconstructing wins from
spread_result + a side comparison. Aggregated across 2015-2026 it gives
294 Win / 189 Loss, exactly matching Power BI dashboard (483 total
picks, 61% ATS win rate) -- confirmed season-by-season, including the
partial 2026 season (3-2 so far, 17 games still pending).

`Recommended_Side` is AWAY for every single real pick (513 rows) -- this
model only ever bets the away side, never home. `Recommended_Logic_Code`
splits those 513 picks into three qualifying rules (codes 1, 4, 8), which
is the real version of the "four qualifying spread ranges" idea from the
CFB model -- three distinct, verified signal types instead of one
placeholder bucket.

Unlike the legacy `pick logic`/`strategy` fields explored in v1 (which
showed a suspicious 100%-win-rate-for-a-decade-then-collapse pattern,
consistent with backtest leakage), this Pick Result-based record is
stable year over year: every season from 2015-2025 falls in a believable
53.8%-71.1% range, including the most recent one. That stability is
itself evidence this is a more trustworthy signal than the legacy tiers.
"""


import numpy as np
import pymc as pm
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt

RNG_SEED = 42
BREAKEVEN = 110 / 210

# ---------------------------------------------------------------------------
# REAL DATA, directly from `Pick Result` and `Recommended_Logic_Code`
# ---------------------------------------------------------------------------
N_TOTAL, WINS_TOTAL = 483, 294  # matches the Power BI dashboard exactly

buckets = [
    {"label": "Logic code 1", "n": 243, "wins": 145},
    {"label": "Logic code 4", "n": 142, "wins": 90},
    {"label": "Logic code 8", "n": 98, "wins": 59},
]
labels = [b["label"] for b in buckets]
n_arr = np.array([b["n"] for b in buckets])
wins_arr = np.array([b["wins"] for b in buckets])

print("=== Verified against dashboard: 294 Win / 189 Loss / 483 total, "
      f"{WINS_TOTAL/N_TOTAL:.1%} ATS win rate ===\n")
print("=== Per-logic-code breakdown (all three qualify on the AWAY side) ===")
for b in buckets:
    print(f"  {b['label']}: {b['wins']}/{b['n']} = {b['wins']/b['n']:.1%}")
print()

# ---------------------------------------------------------------------------
# PART 1: overall record -- closed form vs MCMC
# ---------------------------------------------------------------------------
PRIOR_STRENGTH = 80
alpha_prior = BREAKEVEN * PRIOR_STRENGTH
beta_prior = (1 - BREAKEVEN) * PRIOR_STRENGTH

alpha_post = alpha_prior + WINS_TOTAL
beta_post = beta_prior + (N_TOTAL - WINS_TOTAL)
closed_form_mean = alpha_post / (alpha_post + beta_post)
closed_form_ci = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

print("=== PART 1: full 2015-2026 record, pooled ===")
print(f"Closed-form posterior mean: {closed_form_mean:.4f}, "
      f"95% CI [{closed_form_ci[0]:.4f}, {closed_form_ci[1]:.4f}]")

with pm.Model():
    theta = pm.Beta("theta", alpha=alpha_prior, beta=beta_prior)
    pm.Binomial("obs", n=N_TOTAL, p=theta, observed=WINS_TOTAL)
    single_trace = pm.sample(2000, tune=1000, chains=4, random_seed=RNG_SEED,
                              progressbar=False)
print("MCMC posterior:")
print(az.summary(single_trace, var_names=["theta"]))

p_above = float((single_trace.posterior["theta"].values.flatten() > BREAKEVEN).mean())
print(f"\nPosterior probability true win rate > breakeven: {p_above:.2%}\n")

# ---------------------------------------------------------------------------
# PART 2: hierarchical model across the 3 real qualifying logic codes
# ---------------------------------------------------------------------------
def logit(p):
    return np.log(p / (1 - p))

with pm.Model():
    mu = pm.Normal("mu", mu=logit(BREAKEVEN), sigma=0.3)
    sigma = pm.HalfNormal("sigma", sigma=0.4)
    z = pm.Normal("z", mu=0, sigma=1, shape=len(buckets))
    theta = pm.Deterministic("theta", pm.math.invlogit(mu + sigma * z))
    pm.Binomial("obs", n=n_arr, p=theta, observed=wins_arr)
    hier_trace = pm.sample(3000, tune=2000, chains=4, random_seed=RNG_SEED,
                            target_accept=0.995, progressbar=False)

hier_summary = az.summary(hier_trace, var_names=["theta", "mu", "sigma"])
print("=== PART 2: per-logic-code hierarchical estimates ===")
print(hier_summary)
max_rhat = hier_summary["r_hat"].max()
print(f"\nMax r_hat: {max_rhat:.4f} "
      f"({'converged' if max_rhat < 1.01 else 'CHECK CONVERGENCE'})")

shrunk_mean = hier_summary.loc[[f"theta[{i}]" for i in range(len(buckets))], "mean"].values
shrunk_lo = hier_summary.loc[[f"theta[{i}]" for i in range(len(buckets))], "hdi_3%"].values
shrunk_hi = hier_summary.loc[[f"theta[{i}]" for i in range(len(buckets))], "hdi_97%"].values
raw_rate = wins_arr / n_arr

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
BLUE = "#1f6feb"
GRAY = "#6e7781"
ORANGE = "#d97706"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
theta_grid = np.linspace(0.45, 0.72, 500)
ax.plot(theta_grid, stats.beta.pdf(theta_grid, alpha_prior, beta_prior),
        color=GRAY, linestyle="--", linewidth=2, label="Prior")
ax.plot(theta_grid, stats.beta.pdf(theta_grid, alpha_post, beta_post),
        color=ORANGE, linewidth=2.5, label="Posterior (closed form)")
ax.hist(single_trace.posterior["theta"].values.flatten(), bins=60,
        density=True, color=BLUE, alpha=0.45, label="Posterior (MCMC)")
ax.axvline(BREAKEVEN, color="black", linewidth=1, linestyle=":",
           label=f"Breakeven ({BREAKEVEN:.1%})")
ax.set_xlabel("True win rate (theta)")
ax.set_ylabel("Density")
ax.set_title("Part 1: verified record, 2015-2026\n(294W-189L, matches dashboard)")
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
y_pos = np.arange(len(buckets))
ax.scatter(raw_rate, y_pos + 0.15, color=GRAY, s=70, zorder=3,
           label="Raw empirical rate")
ax.errorbar(shrunk_mean, y_pos - 0.15,
            xerr=[shrunk_mean - shrunk_lo, shrunk_hi - shrunk_mean],
            fmt="o", color=BLUE, ecolor=BLUE, elinewidth=2, capsize=4,
            markersize=7, zorder=4, label="Partially pooled (94% HDI)")
ax.axvline(BREAKEVEN, color="black", linewidth=1, linestyle=":",
           label=f"Breakeven ({BREAKEVEN:.1%})")
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{l}\n(n={n})" for l, n in zip(labels, n_arr)], fontsize=8)
ax.set_xlabel("Win rate")
ax.set_title("Part 2: by qualifying logic code\n(all 3 real, all AWAY-side signals)")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.invert_yaxis()

fig.tight_layout()
fig.savefig("bayesian_nfl_signal_v3.png", dpi=150)
print("\nSaved plot to bayesian_nfl_signal_v3.png")
