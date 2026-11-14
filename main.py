#!/usr/bin/env python3
"""btc-cent-forecast-01: forecast Bitcoin's US-dollar price 240 minutes ahead.

Reads one case at a time from $MAKEFUN_INPUT_DIR (default /input): each
<id>.in holds 4,320 one-minute closes of BTC/USD for the 72 hours up to a
checkpoint. Writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output) with
one forecast: the price expected 240 minutes after the checkpoint.

Method (all fitted inside the case itself, nothing external, no network):
  forecast = checkpoint close * exp(drift), drift clamped to +/-0.02 in log.
  drift = 0.7 * ridge + 0.3 * trend_meanrev, where
  - ridge: a ridge regression trained on pseudo-checkpoints cut inside the
    case (every 30 minutes from 24 h in to 240 min before the end), each
    with the 240-minute-ahead log return as its target and five features:
    log returns over the last 1 h, 4 h, 12 h, 24 h, and the gap between the
    log price and its 24 h mean. Features are standardised, lambda is 3,
    the intercept is not penalised, and the prediction is shrunk 0.7
    towards zero.
  - trend_meanrev: the OLS slope of log price over the last 12 h,
    extrapolated 240 minutes and damped by 0.5, plus 0.3 times the gap
    between the 24 h mean log price and the last log price.

Python 3.13, standard library only, fully deterministic: no randomness,
no wall-clock or environment dependence, so two runs on the same case
write byte-identical output.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HORIZON_MIN = 240
MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999
DRIFT_CLAMP = 0.02
RIDGE_LAMBDA = 3.0
RIDGE_SHRINK = 0.7
TREND_DAMP = 0.5
MEANREV_WEIGHT = 0.3
DRIFT_RIDGE_WEIGHT = 0.7
DRIFT_TREND_WEIGHT = 0.3


def clean_prices(raw):
    """Forward-fill bad ticks (non-number, non-finite, <= 0).

    Returns the cleaned list of floats. Raises ValueError if no good
    close exists at all (nothing to forward-fill from).
    """
    cleaned = []
    last_good = None
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            good = False
        elif not math.isfinite(value) or value <= 0:
            good = False
        else:
            good = True
        if good:
            last_good = float(value)
            cleaned.append(float(value))
        elif last_good is not None:
            cleaned.append(last_good)
        else:
            raise ValueError("no good close to forward-fill from")
    if not cleaned:
        raise ValueError("empty price series")
    return cleaned


def price5(x):
    """Clamp into the scorer's (0, 10_000_000) and round to 5 decimals."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def solve_linear(a, b):
    """Solve a small dense linear system by Gaussian elimination.

    a is a list of n rows (each a list of n floats), b a list of n floats.
    Partial pivoting, deterministic order. Raises ValueError if singular.
    """
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ValueError("singular system")
        m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def features_at(logp, t):
    """Five features of the log-price series at minute index t."""
    mean24 = sum(logp[t - 1440:t + 1]) / 1441.0
    return (
        logp[t] - logp[t - 60],    # 1 h log return
        logp[t] - logp[t - 240],   # 4 h log return
        logp[t] - logp[t - 720],   # 12 h log return
        logp[t] - logp[t - 1440],  # 24 h log return
        mean24 - logp[t],          # gap: 24 h mean log price minus last
    )


def ridge_forecast(logp):
    """Ridge prediction of the 240-minute-ahead log return from the last close.

    Trains on pseudo-checkpoints inside this case only. Returns 0.0 when
    the series is too short to build any training pair.
    """
    n = len(logp)
    last = n - 1
    # Pseudo-checkpoints: every 30 minutes, starting 24 h in (so every
    # feature window is full) and stopping 240 min before the end (so the
    # target return is known from the case: t + 240 must stay in range,
    # i.e. t <= n - 1 - 240).
    starts = range(1440, n - HORIZON_MIN, 30)
    xs = []
    ys = []
    for t in starts:
        xs.append(features_at(logp, t))
        ys.append(logp[t + HORIZON_MIN] - logp[t])
    if not xs:
        return 0.0

    k = len(xs[0])
    means = [sum(x[j] for x in xs) / len(xs) for j in range(k)]
    stds = []
    for j in range(k):
        var = sum((x[j] - means[j]) ** 2 for x in xs) / len(xs)
        stds.append(math.sqrt(var) if var > 0.0 else 1.0)
    z = [[(x[j] - means[j]) / stds[j] for j in range(k)] for x in xs]

    # Normal equations with intercept unpenalised: unknowns are
    # (intercept, beta_0..beta_{k-1}); rows/cols 1..k get +lambda on the
    # diagonal.
    dim = k + 1
    a = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    for row, y in zip(z, ys):
        aug = [1.0] + row
        for i in range(dim):
            b[i] += aug[i] * y
            for j in range(dim):
                a[i][j] += aug[i] * aug[j]
    for j in range(1, dim):
        a[j][j] += RIDGE_LAMBDA
    beta = solve_linear(a, b)

    cur = features_at(logp, last)
    zcur = [(cur[j] - means[j]) / stds[j] for j in range(k)]
    pred = beta[0] + sum(beta[j + 1] * zcur[j] for j in range(k))
    return RIDGE_SHRINK * pred


def trend_meanrev(logp):
    """Trend + mean-reversion drift from the last 12 h / 24 h of log prices."""
    n = len(logp)
    last = logp[n - 1]
    window = min(720, n)
    seg = logp[n - window:n]
    # OLS slope of log price against minutes 0..window-1.
    xm = (window - 1) / 2.0
    ym = sum(seg) / window
    num = sum((i - xm) * (y - ym) for i, y in enumerate(seg))
    den = sum((i - xm) ** 2 for i in range(window))
    slope = num / den if den else 0.0
    trend = TREND_DAMP * slope * HORIZON_MIN
    mean24 = sum(logp[n - 1440:n]) / 1440.0 if n >= 1440 else ym
    return trend + MEANREV_WEIGHT * (mean24 - last)


def forecast(prices):
    """Forecast price 240 minutes after the last close in `prices`."""
    cleaned = clean_prices(prices)
    last = cleaned[-1]
    n = len(cleaned)
    if n < 1440 + HORIZON_MIN + 1:
        # Too short for the full in-case fit: fall back to persistence.
        return price5(last)
    logp = [math.log(p) for p in cleaned]
    ridge = ridge_forecast(logp)
    trend = trend_meanrev(logp)
    drift = DRIFT_RIDGE_WEIGHT * ridge + DRIFT_TREND_WEIGHT * trend
    drift = max(-DRIFT_CLAMP, min(DRIFT_CLAMP, drift))
    return price5(last * math.exp(drift))


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in in_dir.iterdir()
                   if f.is_file() and f.name.endswith(".in"))
    for path in files:
        case_id = path.name[:-len(".in")]
        with path.open("r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        prices = markets[0].get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError(f"{path.name} holds no prices")
        guess = forecast(prices)
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        with (out_dir / f"{case_id}.out").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
