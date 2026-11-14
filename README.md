# btc-cent-forecast-01 — Forecast Bitcoin's price 4 hours ahead

Entry for the make.fun competition `btc-cent-forecast-01`: a program that
reads Bitcoin's 1-minute closes for the 72 hours up to a checkpoint and
guesses the price 240 minutes (4 hours) later. Score: mean absolute error
in cents over the cases, lower is better; a prize needs beating the house
persistence baseline (guess = checkpoint close).

## Files

- `main.py` — the entrypoint (Python 3.13, standard library only, offline,
  deterministic). Run: `python3 main.py`, reading `$MAKEFUN_INPUT_DIR`
  (default `/input`) and writing `$MAKEFUN_OUTPUT_DIR` (default `/output`).
- `LICENSE` — MIT.

## Method

`forecast = checkpoint close * exp(drift)`, with drift clamped to ±0.02 in
log terms. The drift blends two signals, both fitted inside the case
itself — nothing is fetched, nothing is learned from outside the case:

1. **Ridge (weight 0.7).** Pseudo-checkpoints are cut inside the case:
   every 30 minutes from 24 h in to 240 min before the end, each with its
   known 240-minute-ahead log return as the target. Five features per
   point: log returns over the last 1 h, 4 h, 12 h, 24 h, and the gap
   between the log price and its 24 h mean. Features are standardised;
   ridge regression (lambda 3, intercept unpenalised) predicts the next
   240-minute log return; the prediction is shrunk 0.7 towards zero.
2. **Trend + mean reversion (weight 0.3).** The OLS slope of log price
   over the last 12 h, extrapolated 240 minutes and damped by 0.5, plus
   0.3 times the gap between the 24 h mean log price and the last log
   price.

Bad ticks (non-numeric, non-finite, or ≤ 0) are forward-filled from the
last good close; if no good close exists the run errors. The final price
is clamped to [0.00001, 9999999.99999] and rounded to 5 decimals.

## Practice results (kit's six cases, p00–p05)

Actual for p00–p04 is the next case's checkpoint close (checkpoints are
4 h apart); p05 has no actual in the kit. Mean absolute error in cents:

| case | checkpoint | actual | forecast | entry error | baseline error |
|------|-----------:|-------:|---------:|------------:|---------------:|
| p00 | 74764.91526 | 74447.80384 | 74533.58457 | 8,578.07 | 31,711.14 |
| p01 | 74447.80384 | 73680.83594 | 74190.63892 | 50,980.30 | 76,696.79 |
| p02 | 73680.83594 | 73210.50202 | 73763.53334 | 55,303.13 | 47,033.39 |
| p03 | 73210.50202 | 73777.35178 | 73271.80859 | 50,554.32 | 56,684.98 |
| p04 | 73777.35178 | 73545.88170 | 73637.44308 | 9,156.14 | 23,147.01 |
| **mean (5 cases)** | | | | **34,914.39** | **47,054.66** |

The entry beats persistence by 25.8% on the five scored practice cases
(the kit's stated 6-case baseline is 44,521.58 cents). One case, p02, is
worse than baseline: its last-12 h trend pointed up while the price fell
$470. Before settling on these weights I checked alternatives on a
causal walk-forward over 120 pseudo-checkpoints inside the practice
cases (t ≥ 2880, so the ridge has enough training pairs): the chosen
weights scored 31,528 cents vs 42,016 for persistence (85/120 wins),
while the variants that looked better on the five practice cases alone
(pure ridge, lambda 1, shrink 0.9) scored worse there — so the five-case
"improvements" were overfitting and were rejected.

## Determinism and runtime

Two runs on the same six cases produce byte-identical outputs
(verified with `diff -r` and sha256). Runtime is ~0.02 s per case on
this machine, far under the 8 s cap.
