# Indication Quality Metric

At the start of a Bomb Busters mission, each player places one info token on a blue wire on their rack. This document describes the **information gain** metric used to determine the optimal wire to indicate.

## Overview

When a player indicates a wire, they publicly reveal its value and position. Since wires are sorted in ascending order on the rack, this constrains what the remaining hidden wires could be. The **indication quality metric** quantifies how much uncertainty this indication resolves for teammates.

## Information-Theoretic Foundation

### Shannon Entropy

The metric is based on [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)), the standard measure of uncertainty in a probability distribution. For a discrete random variable with probabilities $p_1, p_2, \ldots, p_n$:

$$H = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

Measured in **bits**. Higher entropy means more uncertainty; lower entropy means less.

### Per-Position Entropy

Before any indication, each hidden position on a player's stand has a probability distribution over possible wires. This distribution is constrained by:

- **Sort order**: Wires are ascending by `sort_value` on the rack.
- **Wire pool**: The set of wires that could be at each position (all wires in play minus the player's own wires minus any publicly known wires).
- **Position bounds**: Each hidden slot's wire must have a `sort_value` between its nearest known left and right neighbors.

The **total stand uncertainty** is the sum of per-position entropies:

$$H_{\text{total}} = \sum_{i \in \text{hidden}} H(\text{position}_i)$$

Note: This is an upper bound on the true joint entropy (positions are correlated since they share a pool). However, for ranking indication choices, the sum of marginal entropies is the correct measure: it captures the total uncertainty teammates face when reasoning about each individual position.

### Information Gain

When a player indicates wire $w$ at slot $k$, the info token:

1. Reveals the wire's value at that position (removing it from the hidden set).
2. Tightens the sort-value bounds on neighboring hidden positions.
3. Removes one copy of that wire from the unknown pool.

The **information gain** is:

$$\text{IG}(k) = H_{\text{baseline}} - H_{\text{after}}(k)$$

where:

- $H_{\text{baseline}}$ = total entropy with all positions hidden (no indication).
- $H_{\text{after}}(k)$ = total entropy after indicating slot $k$.

Higher information gain means the indication resolves more uncertainty.

## Computation Algorithm

### Single-Stand Two-Pass Dynamic Programming

The per-position distributions are computed using a two-pass DP over `(wire_type, position)` states. This is a simplified, single-stand version of the backward/forward approach used by the multi-player constraint solver.

**Inputs:**

- **Positions**: $N$ hidden slot constraints (sort-value lower/upper bounds).
- **Wire pool**: $D$ distinct wire types with available counts $c_1, \ldots, c_D$ (all wires in play minus publicly known wires from other stands).

**Key insight:** Since positions are ascending and wire types are sorted by `sort_value`, a composition $(k_1, k_2, \ldots, k_D)$ where $\sum k_i = N$ deterministically assigns the first $k_1$ positions to type 1, the next $k_2$ to type 2, etc. Each composition has a combinatorial weight $\prod_{i=1}^{D} \binom{c_i}{k_i}$, accounting for the number of ways to draw those specific wires from the pool.

**Backward pass** — Computes $f[d][\pi]$, the total combinatorial weight of all valid compositions that fill positions $\pi \ldots N{-}1$ using wire types $d \ldots D{-}1$:

$$f[d][\pi] = \sum_{k=0}^{\text{max\_k}} \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k]$$

Base case: $f[D][N] = 1$ (all types used, all positions filled).

**Forward pass** — Propagates $g[d][\pi]$, the accumulated weight reaching state $(d, \pi)$ from the root ($g[0][0] = 1$). For each transition placing $k$ copies of type $d$ at positions $\pi \ldots \pi{+}k{-}1$:

$$\text{contribution} = g[d][\pi] \cdot \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k]$$

Each position $\pi \ldots \pi{+}k{-}1$ accumulates this contribution for wire type $d$.

**Complexity:** $O(D \times N \times \text{max\_k})$ where $\text{max\_k} \leq 4$ for blue wires. With $D \approx 12$ wire types and $N \approx 10$ positions, this completes in microseconds — no progress bar needed.

### Entropy Calculation

From the accumulated per-position distributions, compute Shannon entropy for each hidden position:

$$H(\text{pos}_i) = -\sum_{w} p_{i,w} \log_2(p_{i,w})$$

where $p_{i,w} = \text{count}_{i,w} \, / \, \text{total\_weight}$.

Sum across all hidden positions to get total entropy.

### Per-Indication Evaluation

For each candidate indication (each blue wire on the stand):

1. Temporarily mark the slot as INFO_REVEALED with the wire's gameplay value.
2. Recompute sort-value bounds for remaining hidden positions (neighbors now have tighter constraints from the info token).
3. Remove one copy of the indicated wire from the pool.
4. Run the two-pass DP on the remaining hidden positions with the updated pool and bounds.
5. Compute the remaining entropy.
6. Information gain = baseline entropy minus remaining entropy.

## Interpreting the Results

### Information Gain (bits)

The primary metric. A **bit** is one binary digit of information — the amount learned from a single yes/no question. The key property of the logarithmic scale:

> **Each bit halves the number of equally-likely possibilities.**

If a position has 8 equally-likely wire types, its entropy is $\log_2(8) = 3$ bits. An indication that eliminates half the options reduces entropy by 1 bit. More precisely, $N$ bits of information gain means the number of possible wire arrangements was reduced by a factor of $2^N$:

| Information Gain | Possibility Reduction | Equivalent |
|------------------|-----------------------|------------|
| 1 bit | 2× fewer | 1 yes/no question answered |
| 2 bits | 4× fewer | 2 yes/no questions answered |
| 4 bits | 16× fewer | 4 yes/no questions answered |
| 6 bits | 64× fewer | 6 yes/no questions answered |
| 8 bits | 256× fewer | 8 yes/no questions answered |

**Important:** Because the scale is logarithmic, equal *differences* in bits correspond to equal *ratios* of possibility reduction. Going from 4 to 6 bits is the same multiplicative improvement (4×) as going from 6 to 8 bits — not the same additive improvement. An indication with 8 bits of IG does not resolve "twice as much" as one with 4 bits — it resolves **16 times** as much (the difference is 4 bits = $2^4 = 16\times$).

Since the metric sums per-position entropies across all hidden positions on the stand, a total information gain of 6 bits might mean (for example) that 3 positions each had their uncertainty halved by 2 bits, or that 1 position's uncertainty dropped by 6 bits while others were unaffected. The total captures the aggregate information revealed.

Typical values for a 10-position stand with 12 distinct blue wire types:

| Range | Interpretation |
|-------|---------------|
| 0.0 - 0.5 bits | Minimal. The indication barely constrains the stand (< 2× reduction). |
| 0.5 - 1.5 bits | Moderate. Useful constraint for teammates (2×–3× reduction). |
| 1.5 - 3.0 bits | Strong. Significantly narrows possibilities (3×–8× reduction). |
| 3.0+ bits | Excellent. Usually from cluster-edge indications (8×+ reduction). |

These ranges are approximate and depend on stand size and pool composition.

### Uncertainty Resolved (%)

A normalized, linear view of information gain: what fraction of the total baseline uncertainty does this indication resolve?

$$\text{Uncertainty Resolved} = \frac{\text{IG}}{H_{\text{baseline}}} \times 100\%$$

Unlike bits, this metric **is linear and directly comparable**: 30% resolved is twice as informative as 15% resolved. This makes it the easier metric for at-a-glance comparisons between indication choices.

For example, if the baseline entropy is 25 bits and an indication gains 5 bits, that's 20% resolved — one-fifth of the total uncertainty about the stand has been eliminated.

#### When to use which metric

- **Bits** are meaningful in absolute terms: 3 bits always means an 8× reduction in possibilities, regardless of stand size. Use bits to understand the raw power of an indication.
- **Percentage** is meaningful in relative terms: it tells you what fraction of the *total problem* this indication solves. Use percentage to compare indications across different stand sizes or pool compositions, or to quickly gauge "how good" an indication is.

## What Influences the Metric

### Cluster-Edge Indications

Stands with clusters of similar values benefit most from indicating at the cluster boundary. For example, with stand `1 1 2 2 2 3 3 4 8 10`:

- Indicating `4` at slot H tells teammates: "everything to my left is 4 or less." This massively constrains 7 positions.
- Indicating `8` at slot I gives less information since it only constrains 1 position to the right.

### Expected vs. Surprising Positions

- Indicating `1` at slot A (the leftmost position) only confirms you have a 1. Others already expected the lowest value at position A.
- Indicating `4` at slot H (unusually high position for value 4) is surprising and informative: it reveals the stand is heavily skewed toward low values.

### Colored Wire Safety

When red or yellow wires are in the pool, their positions are uncertain. Indicating near a dangerous wire's sort value helps teammates identify safe zones:

- If red wire 7.5 is in play and you indicate `8` at slot F, teammates know slots A-E cannot contain anything with `sort_value > 8.0`, and all positions from F onward are `>= 8.0`, ruling out red 7.5 from slots G+.
- The entropy metric captures this naturally: positions where red wires are ruled out have lower entropy.

### Previous Indications

When other players have already indicated (clockwise from captain), their publicly visible info tokens:

1. Remove their indicated wire from the pool (it is known to be on their stand).
2. Do not directly constrain the current player's stand, but the pool reduction can subtly shift which indications are most valuable.

## Limitations

- **Marginal vs. joint entropy**: The metric sums per-position marginal entropies, which is an upper bound on the true joint entropy. For ranking purposes, this is appropriate since teammates reason about positions independently when deciding where to cut.
- **Generic observer model**: The metric uses the full complement pool (all wires in play minus the indicating player's wires). Individual observers have smaller pools (they also know their own wires), which means the actual entropy reduction they experience may differ slightly. The relative ranking of indication choices is robust to this.
- **Uncertain (X-of-Y) wire groups**: Currently handled by including all candidates in the pool, which slightly overestimates uncertainty. The ranking remains valid.
