# NFL & CFB Betting Models: Pipeline, Modeling, and Bayesian Validation

Prepared by Jeb Pittard. Two independent against-the-spread (ATS) models, each running full pipelines from live data ingestion through model scoring to statistical validation of the result.

---

## 1. CFB Spread Model

### Data ingestion (Databricks / PySpark)

Python handles the API ingestion layer: `requests.get()` calls to the College Football Data API, with `RequestException` error handling and retry logic. JSON responses are parsed via a `transform_game_data()` function that maps camelCase API fields to the target Delta table schema, with null safe type casting (`safe_float()`). Data moves from Pandas to Spark with `pd.DataFrame` to `spark.createDataFrame` to `write.mode("append").saveAsTable`.

Betting lines are processed with `scipy.stats.mode` to select the consensus spread across sportsbooks, and a `createOrReplaceTempView` plus `MERGE INTO` pattern updates spreads in bulk as new lines come in.

### Modeling approach

OLS and logistic regression, with era based splits to account for the post 2021 NIL rule change. Backtested on 13,624 games (2021 to 2025).

### Results

1,327 selected picks, 57.7% ATS win rate, +7.7% EV per game.

### Calibration bug caught

An early "Edge" field was built to flag high value bets. When checked against actual outcomes it wasn't well calibrated. It looked good in aggregate but didn't reflect true win probability, so it was pulled from the model rather than shipped, until it could be fixed properly.

---

## 2. NFL ATS Model

### Data ingestion (Databricks / PySpark)

The NFL pipeline uses Python to fetch `games.csv` from the nflverse GitHub via `pd.read_csv()`, filters to relevant seasons and game types, and writes into Delta tables. A separate Python notebook calls The Odds API for live spreads, computes the median across US bookmakers, converts UTC kickoff times to Eastern via `FROM_UTC_TIMESTAMP`, and updates unplayed games across all four tables using `MERGE INTO`.

### Modeling approach

Model runs in production via Power BI/DAX (`NFL_ATS_FINAL_HOME_PYSPARK`). The model always recommends the away side, and qualifies picks into three distinct signal types ("logic codes") rather than one blanket rule.

### Results

Verified directly against the production dashboard: 294 wins, 189 losses, 483 decided picks, 2015 through 2026, a 60.9% ATS win rate. Matches season by season, including the partial 2026 season.

By logic code:
- Logic code 1: 145 of 243, 59.7%
- Logic code 4: 90 of 142, 63.4%
- Logic code 8: 59 of 98, 60.2%

### Bayesian / MCMC validation (PyMC)

Built a Beta-Binomial posterior over the full 294-189 record, solved both in closed form and via MCMC (NUTS sampler), to confirm the two methods agree before trusting MCMC on the harder problem below.

- Closed-form posterior mean: 59.7%
- 95% credible interval: 55.6% to 63.7%
- MCMC posterior (r-hat 1.0, clean convergence): matches closed form
- Posterior probability the true win rate beats the 52.4% breakeven needed at standard -110 odds: 99.98%

Then built a hierarchical (partial pooling) model across the three logic codes, using a non-centered parameterization to avoid the "funnel" geometry that causes hierarchical models to fail. Initial run produced 5 divergent transitions; raising `target_accept` to 0.995 and increasing tuning and draw counts produced a clean run with r-hat of 1.0 across all parameters.

### Data quality finding: legacy field leakage

While validating the model, an older set of fields in the same dataset (not the ones the production model currently uses) showed a near 100% win rate for ten straight seasons that collapsed to a coin flip in the most recent season. That pattern, a decade of implausibly perfect results followed by a sudden cliff in the newest out-of-sample data, is a textbook signature of look-ahead bias or leakage in a historical backtest, not a real edge.

Tracing the issue down to the correct, authoritative outcome field (`Pick Result`), and verifying it three independent ways against raw scores and spreads, confirmed that the model's actual current signal is the stable, modest ~60% result reported above, not the inflated legacy number.

---

## 3. Tech stack summary

Python (pandas, NumPy, scipy), PySpark, Databricks Delta Lake, REST API ingestion (College Football Data API, The Odds API, nflverse), Power BI/DAX for production scoring, PyMC/ArviZ for Bayesian inference and MCMC diagnostics.

---

## 4. Known limitations / next steps

- One pricing field (`Sum of PM Edge %`) currently appears to equal each logic code's own realized win rate exactly, which suggests it may be a descriptive backtest statistic rather than an independent forward-looking prediction. Worth confirming before relying on it.
- An older recommendation field disagreed with the current production pick on a meaningful share of overlapping games during validation. Being investigated before either field is fully trusted going forward.
