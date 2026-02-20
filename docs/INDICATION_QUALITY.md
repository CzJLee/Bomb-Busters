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

$$f[d][\pi] = \sum_{k=0}^{k_{\max}} \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k]$$

Base case: $f[D][N] = 1$ (all types used, all positions filled).

**Forward pass** — Propagates $g[d][\pi]$, the accumulated weight reaching state $(d, \pi)$ from the root ($g[0][0] = 1$). For each transition placing $k$ copies of type $d$ at positions $\pi \ldots \pi{+}k{-}1$:

$$\text{contribution} = g[d][\pi] \cdot \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k]$$

Each position $\pi \ldots \pi{+}k{-}1$ accumulates this contribution for wire type $d$.

**Complexity:** $O(D \times N \times k_{\max})$ where $k_{\max} \leq 4$ for blue wires. With $D \approx 12$ wire types and $N \approx 10$ positions, this completes in microseconds — no progress bar needed.

### Entropy Calculation

From the accumulated per-position distributions, compute Shannon entropy for each hidden position:

$$H(\text{pos}_i) = -\sum_{w} p_{i,w} \log_2(p_{i,w})$$

where $p_{i,w}$ is the normalized weight of wire $w$ at position $i$.

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

### Uncertainty Resolved (%)

A normalized, linear view of information gain: what fraction of the total baseline uncertainty does this indication resolve?

$$\text{Uncertainty Resolved} = \frac{\text{IG}}{H_{\text{baseline}}} \times 100\%$$

Unlike bits, this metric **is linear and directly comparable**: 30% resolved is twice as informative as 15% resolved. This makes it the easier metric for at-a-glance comparisons between indication choices.

For example, if the baseline entropy is 25 bits and an indication gains 5 bits, that's 20% resolved — one-fifth of the total uncertainty about the stand has been eliminated.

#### Empirical benchmarks

The following thresholds were derived from 500 random 5-player games (blue wires 1–12, 0–4 yellow, 0–3 red). Mean uncertainty resolved per indication is approximately 20%, with a standard deviation of approximately 7%.

| Range | Interpretation | Statistical context |
|-------|---------------|---------------------|
| < 13% | Poor. The indication barely constrains the stand. | Below mean − 1σ |
| 13% – 20% | Moderate. Useful constraint for teammates. | Between mean − 1σ and mean |
| 20% – 25% | Good. Noticeably narrows possibilities. | Between mean and mean + 1σ |
| ≥ 25% | Excellent. Usually from cluster-edge indications. | Above mean + 1σ |

These thresholds match the color coding used in the terminal display (red, yellow, blue, green respectively).

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

## Parity Indication (Even/Odd Tokens)

Some missions use even/odd tokens instead of standard info tokens. With parity indication, the indicated slot reveals only whether the wire is **even** (2, 4, 6, 8, 10, 12) or **odd** (1, 3, 5, 7, 9, 11) — not its exact value.

### Key Differences from Standard Indication

| Aspect | Standard Info Token | Even/Odd Parity Token |
|--------|--------------------|-----------------------|
| Value revealed | Exact number (e.g., "7") | Only parity (even or odd) |
| Slot state | Becomes INFO_REVEALED | Stays HIDDEN |
| Wire in pool | Removed (identity known) | Stays in pool (identity unknown) |
| Neighbor bounds | Tightened by known value | NOT tightened (value unknown) |
| New constraint | None (slot is resolved) | `required_parity` on the slot |

### How Parity Information Gain is Computed

The parity analysis uses the same two-pass DP infrastructure as standard indication, but with different indication semantics:

1. The indicated slot stays HIDDEN (not converted to INFO_REVEALED).
2. The wire is NOT removed from the pool (its exact value is unknown).
3. A `required_parity` constraint is added to the slot's `PositionConstraint`, which filters out wires of the wrong parity via `wire_fits()`.
4. Neighboring slots do NOT get tighter sort-value bounds.

The information gain is computed identically: baseline entropy minus remaining entropy after the parity constraint is applied.

### Expected Information Gain

Parity indication reveals strictly less information than standard indication (it partitions possibilities into two groups instead of identifying the exact wire). Typical parity information gains are 1-2 bits per indication, compared to 4-8 bits for standard indication. The ranking of choices still follows the same principle: indicate where the parity constraint eliminates the most uncertainty.

Even/odd tokens are never used in missions with yellow wires. The parity analysis functions (`rank_indications_parity()`, `print_indication_analysis_parity()`) only consider blue wires.

#### Empirical benchmarks (parity)

The following thresholds were derived from 500 random 5-player games (blue wires 1–12, all choices per player, n=24,000). Mean uncertainty resolved per parity indication is approximately 5.0%, with a standard deviation of approximately 1.5%.

| Range | Interpretation | Statistical context |
|-------|---------------|---------------------|
| < 3.5% | Poor. The parity constraint barely narrows possibilities. | Below mean − 1σ |
| 3.5% – 5% | Moderate. Useful partitioning for teammates. | Between mean − 1σ and mean |
| 5% – 7% | Good. Noticeably reduces stand uncertainty. | Between mean and mean + 1σ |
| ≥ 7% | Excellent. Strong partitioning effect. | Above mean + 1σ |

These thresholds match the color coding used in the parity terminal display (red, yellow, blue, green respectively).

## Multiplicity Indication (x1/x2/x3 Tokens)

Some missions use x1/x2/x3 tokens instead of standard info tokens. With multiplicity indication, the indicated slot reveals **how many copies of that wire's value exist on the entire stand** (including already-cut wires) — but not the value itself.

### Key Differences from Standard Indication

| Aspect | Standard Info Token | x1/x2/x3 Multiplicity Token |
|--------|--------------------|-----------------------------|
| Value revealed | Exact number (e.g., "7") | NOT revealed |
| Multiplicity revealed | Not directly | xM: how many copies of this value on stand |
| Slot state | Becomes INFO_REVEALED | Stays HIDDEN |
| Wire in pool | Removed (identity known) | Stays in pool (identity unknown) |
| Neighbor bounds | Tightened by known value | NOT tightened (value unknown) |
| New constraint | None (slot is resolved) | Count constraint: exactly M copies of the (unknown) value on the stand |

### How Multiplicity Information Gain is Computed

Multiplicity indication is more complex than standard or parity indication because the constraint is **global** (applies to the count of a value across the entire stand) rather than **local** (constraining a single position). Additionally, the observer does not know which value is at the indicated position — they only see the xM token.

#### Count-Tracking Two-Pass DP

The core computation uses an extended version of the standard two-pass DP that tracks how many copies of a target wire value have been placed. The state space is `(wire_type, position, target_count)`:

**Backward pass** — Computes $f[d][\pi][c]$: the total combinatorial weight of valid compositions that fill positions $\pi \ldots N{-}1$ using wire types $d \ldots D{-}1$, with exactly $c$ more copies of the target gameplay value still needed:

$$f[d][\pi][c] = \sum_{k=0}^{k_{\max}} \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k][\text{new\_c}]$$

where $\text{new\_c} = c - k$ if wire type $d$ has the target gameplay value, or $c$ otherwise.

Base case: $f[D][N][0] = 1$ (all types used, all positions filled, exact target count met).

**Forward pass** — Propagates $g[d][\pi][c]$ with the same count tracking, distributing per-position wire counts weighted by both forward and backward values.

**Indicated position handling**: At the indicated position, only the target wire type (or non-target types if $k > 0$ target copies are still needed) can be placed. Non-target wire types have their `max_run` set to 0 at the indicated position, preventing invalid placements.

**Complexity:** $O(D \times N \times k_{\max} \times (M+1))$ where $M$ is the required multiplicity (1, 2, or 3). This adds a factor of at most 4 to the standard DP, keeping computation in the microsecond range.

#### Summing Over Unknown Values

The observer sees only the xM token, not the value at the indicated position. To correctly compute information gain, the analysis must consider **all possible gameplay values** that could be at the indicated position.

For each candidate indication (each blue wire on the stand), the analysis:

1. Determines the multiplicity $M$ (total copies of this value on the stand, including cut wires).
2. For each possible gameplay value $V'$ that could fit at the indicated position (based on sort-value bounds):
   a. Computes how many hidden copies of $V'$ must exist on the stand: $M_{\text{hidden}} = M - \text{known\_count}(V')$.
   b. Skips $V'$ if $M_{\text{hidden}} < 0$ (impossible) or if the pool doesn't have enough copies.
   c. Runs the count-tracking DP conditioned on the target value being $V'$ with exactly $M_{\text{hidden}}$ hidden copies needed.
3. **Merges** the distributions from all candidate values by summing their per-position counters and total weights.
4. Computes entropy from the merged distribution.

This summation correctly models the observer's uncertainty: after seeing xM, they don't know which value is at the position, so the remaining uncertainty is a weighted average over all compatible values.

$$H_{\text{after}} = H\left(\text{merge}\left(\bigcup_{V' \in \text{possible}} \text{DP}(V', M_{\text{hidden}}(V'))\right)\right)$$

### Expected Information Gain

Multiplicity indication reveals less information than either standard or parity indication. The xM token constrains the global count of an unknown value, which is a weaker constraint than knowing the value (standard) or even its parity (even/odd). Typical multiplicity information gains are 0.5-1.5 bits per indication, compared to 4-8 bits for standard and 1-2 bits for parity.

Multiplicity tokens can be used with yellow wires (unlike even/odd tokens). The multiplicity analysis functions (`rank_indications_multiplicity()`, `print_indication_analysis_multiplicity()`) consider only blue wires when computing information gain.

#### Negative information gain

Unlike standard and parity indication, multiplicity indication can produce **negative information gain** for some choices. This occurs because the metric uses sum-of-marginals entropy (an upper bound on joint entropy), and the multiplicity constraint can make some per-position marginal distributions *more uniform* even though the total number of valid compositions decreases. Approximately 22% of all multiplicity indication choices (not just the best) have negative IG.

This is a measurement artifact, not a real increase in uncertainty. The multiplicity constraint always reduces the set of valid compositions (true joint entropy always decreases). The sum-of-marginals metric simply overestimates entropy in some cases. For ranking purposes, choices with negative IG are genuinely worse than those with positive IG — they provide the least useful information to teammates.

#### Empirical benchmarks (multiplicity)

The following thresholds were derived from 500 random 5-player games (blue wires 1–12, all choices per player, n=24,000). Due to the heavily skewed distribution (22% negative values), percentile-based thresholds are used instead of mean±σ.

| Range | Interpretation |
|-------|---------------|
| < 1% | Poor. Minimal constraint on the stand. Includes negative IG. |
| 1% – 3.5% | Moderate. Some useful information for teammates. |
| 3.5% – 5% | Good. Noticeably constrains possibilities. |
| ≥ 5% | Excellent. Strong multiplicity constraint. |

These thresholds match the color coding used in the multiplicity terminal display (red, yellow, blue, green respectively).

### Comparison of Indication Types

| Metric | Standard | Parity (E/O) | Multiplicity (xM) |
|--------|----------|-------------|-------------------|
| Best indication mean | ~27% resolved | ~8% resolved | ~6% resolved |
| All choices mean | ~19% resolved | ~5% resolved | ~2.4% resolved |
| Typical IG range | 4–8 bits | 1–2 bits | 0.5–1.5 bits |
| Can have negative IG? | No | No | Yes (~22% of choices) |
| Works with yellow wires? | Yes | No | Yes |

## Limitations

- **Marginal vs. joint entropy**: The metric sums per-position marginal entropies, which is an upper bound on the true joint entropy. For ranking purposes, this is appropriate since teammates reason about positions independently when deciding where to cut. This approximation is the cause of negative IG values in multiplicity indication.
- **Generic observer model**: The metric uses the full complement pool (all wires in play minus the indicating player's wires). Individual observers have smaller pools (they also know their own wires), which means the actual entropy reduction they experience may differ slightly. The relative ranking of indication choices is robust to this.
- **Uncertain (X-of-Y) wire groups**: Currently handled by including all candidates in the pool, which slightly overestimates uncertainty. The ranking remains valid.
