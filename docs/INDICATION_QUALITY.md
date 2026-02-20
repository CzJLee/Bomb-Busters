# Indication Quality Metric

At the start of a Bomb Busters mission, each player places one indicator token on a wire on their rack. This document describes the **information gain** metric used to determine the optimal wire to indicate, and how it adapts to different indicator token types.

## Overview

When a player indicates a wire, they publicly reveal some information about it. Since wires are sorted in ascending order on the rack, this constrains what the remaining hidden wires could be. The **indication quality metric** quantifies how much uncertainty this indication resolves for teammates.

Three indicator token types are supported, each revealing different information:

| Token type | What is revealed | Wires eligible |
|------------|-----------------|----------------|
| **Standard info** | Exact value (e.g., "7") | Blue only |
| **Even/Odd parity** | Whether the value is even or odd | Blue only |
| **x1/x2/x3 multiplicity** | How many copies of the value exist on the stand | Blue and yellow |

## Information-Theoretic Foundation

### Shannon Entropy

The metric is based on [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)), the standard measure of uncertainty in a probability distribution. For a random variable $X$ with outcomes $x_1, \ldots, x_n$ and probabilities $p_1, \ldots, p_n$:

$$H(X) = -\sum_{i=1}^{n} p_i \log_2 p_i$$

Measured in **bits**. Higher entropy means more uncertainty; lower entropy means less.

### Joint Entropy

Each valid wire arrangement on a stand is a **composition** — a specific assignment of wires to positions. The wire pool, sort-order constraints, and position bounds together define a set of valid compositions, each with a combinatorial weight (see [Combinatorial Weights](#combinatorial-weights)).

The **joint entropy** of the stand measures the uncertainty over the full joint distribution of all hidden positions simultaneously:

$$H = \log_2 W - \frac{1}{W} \sum_{c} w_c \log_2 w_c$$

where $W = \sum_c w_c$ is the total weight across all valid compositions $c$, and $w_c$ is the weight of composition $c$.

This is equivalent to the standard entropy formula applied to the normalized distribution $p_c = w_c / W$, but expressed in a form that avoids explicitly normalizing each weight.

### Information Gain

When a player indicates wire $w$ at slot $k$, the indication constrains the set of valid compositions. The **information gain** is:

$$\mathrm{IG}(k) = H_{\text{baseline}} - H_{\text{after}}(k)$$

where:

- $H_{\text{baseline}}$ = joint entropy with all positions hidden (no indication).
- $H_{\text{after}}(k)$ = joint entropy after the indication at slot $k$.

Joint entropy correctly captures inter-position correlations (positions share a pool, so they are not independent). Unlike sum-of-marginal-entropy approaches — which can produce spurious negative values because overcounting correlations inflates the baseline — joint entropy yields **non-negative information gain** for all indication types in practice.

Higher information gain means the indication resolves more uncertainty.

## Computation Algorithm

### Composition-Based Dynamic Programming

The entropy computation uses a backward DP over `(wire_type, position)` states. This is a simplified, single-stand version of the composition-based approach used by the multi-player constraint solver (see [EXACT_SOLVER.md](EXACT_SOLVER.md)).

**Inputs:**

- **Positions**: $N$ hidden slot constraints (sort-value lower/upper bounds, optional parity constraint).
- **Wire pool**: $D$ distinct wire types with available counts $c_1, \ldots, c_D$ (all wires in play minus publicly known wires from other stands).

**Key insight:** Since positions are ascending and wire types are sorted by `sort_value`, a composition $(k_1, k_2, \ldots, k_D)$ where $\sum k_i = N$ deterministically assigns the first $k_1$ positions to type 1, the next $k_2$ to type 2, etc.

### Combinatorial Weights

Each composition has a weight equal to the product of binomial coefficients:

$$w_c = \prod_{d=1}^{D} \binom{c_d}{k_d}$$

This accounts for the number of ways to choose which $k_d$ copies of wire type $d$ are placed on this stand out of $c_d$ available copies. The weighting implements the [multivariate hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution#Multivariate_hypergeometric_distribution) — the correct probability model for dealing without replacement from a finite pool.

### Entropy Semiring DP

Computing joint entropy naively would require enumerating every valid composition and summing $w_c \log_2 w_c$ — exponential in the worst case. Instead, we use the **entropy semiring** technique (Li & Eisner, "First- and Second-Order Expectation Semirings", EMNLP 2009), which computes the joint entropy using the same DP structure at the same polynomial cost.

The DP tracks two quantities per state:

| Symbol | Meaning |
|--------|---------|
| $f[d][\pi]$ | Total combinatorial weight of valid completions for positions $\pi \ldots N{-}1$ using wire types $d \ldots D{-}1$ |
| $e[d][\pi]$ | $\sum_c w_c \cdot \log_2 w_c$ summed over the same completions |

**Recurrence for** $f$ (standard weight DP):

$$f[d][\pi] = \sum_{k=0}^{k_{\max}} \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k]$$

Base case: $f[D][N] = 1$.

**Recurrence for** $e$ (entropy semiring):

$$e[d][\pi] = \sum_{k=0}^{k_{\max}} \binom{c_d}{k} \left( \log_2 \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k] + e[d{+}1][\pi{+}k] \right)$$

Base case: $e[D][N] = 0$.

**Derivation:** Each composition's weight factors as $w_c = \binom{c_d}{k} \cdot w_{\text{rest}}$. The log-weight sum decomposes:

$$w_c \log_2 w_c = \binom{c_d}{k} \cdot w_{\text{rest}} \cdot \left( \log_2 \binom{c_d}{k} + \log_2 w_{\text{rest}} \right)$$

Summing over all completions and grouping by the choice of $k$ at level $d$ yields the recurrence above.

**Joint entropy** from the root:

$$H = \log_2 f[0][0] - \frac{e[0][0]}{f[0][0]}$$

Only the backward pass is needed — no forward pass — since the scalar entropy requires only the root values $f[0][0]$ and $e[0][0]$.

**Complexity:** $O(D \times N \times k_{\max})$ where $k_{\max} \le 4$ for blue wires. With $D \approx 12$ wire types and $N \approx 10$ positions, this completes in microseconds.

### Per-Indication Evaluation

How each candidate indication is evaluated depends on the token type (described in detail in the sections below). The general pattern is:

1. Compute baseline joint entropy $H_{\text{baseline}}$ with all positions hidden.
2. For each candidate, simulate the indication effect on constraints and/or pool.
3. Compute $H_{\text{after}}$ via the entropy semiring DP.
4. Information gain = $H_{\text{baseline}} - H_{\text{after}}$.

## Token Type Details

### Standard Info Tokens

Standard info tokens reveal the **exact value** of a blue wire at the indicated position. This is the strongest form of indication.

**Indication effect:**

1. The indicated slot becomes INFO_REVEALED with the wire's gameplay value.
2. Sort-value bounds on neighboring hidden positions are tightened (neighbors can use the revealed value as a boundary).
3. One copy of the indicated wire is removed from the pool.

The indicated slot is excluded from the post-indication DP (its identity is fully resolved), and the pool shrinks by one wire. Both effects reduce entropy.

**Eligible wires:** Blue wires only.

**Deduplication:** Each `(slot_index, sort_value)` pair is evaluated at most once — duplicate wires at the same position produce identical results.

### Even/Odd Parity Tokens

Parity tokens reveal only whether the wire is **even** (2, 4, 6, 8, 10, 12) or **odd** (1, 3, 5, 7, 9, 11). This is a weaker form of indication.

**Indication effect:**

1. The indicated slot stays HIDDEN (not converted to INFO_REVEALED).
2. The wire is NOT removed from the pool (its exact value is unknown).
3. A `required_parity` constraint is added to the slot's position constraint, filtering out wires of the wrong parity via `wire_fits()`.
4. Neighboring slots do NOT get tighter sort-value bounds (the value is unknown).

Because the slot stays in the DP and the pool is unchanged, the only source of entropy reduction is the parity constraint eliminating roughly half the candidate wire types at that position.

**Eligible wires:** Blue wires only. Even/odd tokens are never used in missions with yellow wires.

**Deduplication:** Each `(slot_index, parity)` pair is evaluated at most once — different wires of the same parity at the same position produce the same constraint.

### x1/x2/x3 Multiplicity Tokens

Multiplicity tokens reveal **how many copies of the indicated wire's value exist on the entire stand** (including already-cut wires) — but not the value itself. This is the most complex indication type.

**Indication effect:**

1. The indicated slot stays HIDDEN (not converted to INFO_REVEALED).
2. The wire is NOT removed from the pool.
3. Neighboring slots do NOT get tighter sort-value bounds.
4. A **global count constraint** is imposed: the (unknown) value at the indicated position must appear exactly $M$ times total on the stand.

**Eligible wires:** Blue and yellow wires. Red wires cannot be indicated with multiplicity tokens.

#### Mathematical Formulation

The observer sees "x$M$ at position $[G]$" but doesn't know which value $V$ is at position $G$. The post-indication entropy must account for this uncertainty by **summing over all possible values**.

Let $\mathcal{V}$ be the set of gameplay values that could fit at the indicated position (based on sort-value bounds). For each candidate value $V' \in \mathcal{V}$:

- $M_{\text{known}}(V')$: copies of $V'$ already visible on the stand (CUT or INFO_REVEALED).
- $M_{\text{hidden}}(V') = M - M_{\text{known}}(V')$: required hidden copies. Skip $V'$ if negative or exceeds pool availability.

Each value $V'$ produces a **disjoint** set of valid compositions (they are mutually exclusive because the wire at the indicated position can only be one value). The per-value entropy semiring DP uses an extended state space $f[d][\pi][c]$ and $e[d][\pi][c]$ where $c$ tracks how many more target-value wires must still be placed:

$$f[d][\pi][c] = \sum_{k=0}^{k_{\max}} \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k][c']$$

where $c' = c - k$ if wire type $d$ has gameplay value $V'$, or $c' = c$ otherwise. Non-target wire types are blocked from the indicated position via `max_run`.

Base case: $f[D][N][0] = 1$. The $e$ recurrence is the same entropy semiring extension as before, with the additional $[c]$ dimension.

The per-value DP yields $(H_{V'}, W_{V'})$ — the joint entropy and total weight conditioned on $V'$. Since the composition sets are disjoint, the merged entropy is:

$$H_{\text{after}} = \log_2 W_{\text{total}} - \frac{1}{W_{\text{total}}} \sum_{V'} S_{V'}$$

where $W_{\text{total}} = \sum_{V'} W_{V'}$ and $S_{V'} = \sum_{c \in V'} w_c \log_2 w_c$ is recovered from each sub-call via:

$$S_{V'} = W_{V'} \cdot (\log_2 W_{V'} - H_{V'})$$

**Complexity:** $O(D \times N \times k_{\max} \times (M+1) \times |\mathcal{V}|)$. With typical values $M \le 3$ and $|\mathcal{V}| \le 12$, this remains fast.

**Deduplication:** Each `(slot_index, multiplicity)` pair is evaluated at most once — different wires with the same total count at the same position produce the same xM token.

### Comparison of Indication Types

| Aspect | Standard | Parity (E/O) | Multiplicity (xM) |
|--------|----------|-------------|-------------------|
| Value revealed | Exact number | Even or odd | Not revealed |
| Slot state after | INFO_REVEALED | Stays HIDDEN | Stays HIDDEN |
| Wire removed from pool | Yes | No | No |
| Neighbor bounds tightened | Yes | No | No |
| Constraint type | Slot resolved | Local (parity filter) | Global (count across stand) |
| Works with yellow | No | No | Yes |
| Typical IG range | 4-8 bits | 1-2 bits | 0.5-1.5 bits |

## Interpreting the Results

### Information Gain (bits)

The primary metric. A **bit** is one binary digit of information — the amount learned from a single yes/no question. The key property of the logarithmic scale:

> **Each bit halves the number of equally-likely possibilities.**

If a stand has $2^{20}$ equally-likely wire arrangements, its entropy is 20 bits. An indication gaining 6 bits reduces possibilities by a factor of $2^6 = 64$.

| Information Gain | Possibility Reduction | Equivalent |
|------------------|-----------------------|------------|
| 1 bit | 2x fewer | 1 yes/no question answered |
| 2 bits | 4x fewer | 2 yes/no questions answered |
| 4 bits | 16x fewer | 4 yes/no questions answered |
| 6 bits | 64x fewer | 6 yes/no questions answered |
| 8 bits | 256x fewer | 8 yes/no questions answered |

**Important:** Because the scale is logarithmic, equal *differences* in bits correspond to equal *ratios* of possibility reduction. Going from 4 to 6 bits is the same multiplicative improvement (4x) as going from 6 to 8 bits — not the same additive improvement. An indication with 8 bits of IG does not resolve "twice as much" as one with 4 bits — it resolves **16 times** as much (the difference is 4 bits, so $2^4 = 16$ times the reduction).

### Uncertainty Resolved (%)

A normalized, linear view of information gain: what fraction of the total baseline uncertainty does this indication resolve?

$$\text{Uncertainty Resolved} = \frac{\mathrm{IG}}{H_{\text{baseline}}} \times 100\%$$

Unlike bits, this metric **is linear and directly comparable**: 30% resolved is twice as informative as 15% resolved. This makes it the easier metric for at-a-glance comparisons between indication choices.

For example, if the baseline entropy is 25 bits and an indication gains 5 bits, that's 20% resolved — one-fifth of the total uncertainty about the stand has been eliminated.

#### Empirical benchmarks (standard info tokens)

The following thresholds were derived from 500 random 5-player games (blue wires 1-12, 0-4 yellow, 0-3 red). Mean uncertainty resolved per indication is approximately 20%, with a standard deviation of approximately 7%.

| Range | Interpretation | Statistical context |
|-------|---------------|---------------------|
| < 13% | Poor. The indication barely constrains the stand. | Below mean - 1 sigma |
| 13% - 20% | Moderate. Useful constraint for teammates. | Between mean - 1 sigma and mean |
| 20% - 25% | Good. Noticeably narrows possibilities. | Between mean and mean + 1 sigma |
| >= 25% | Excellent. Usually from cluster-edge indications. | Above mean + 1 sigma |

These thresholds match the color coding used in the terminal display (red, yellow, blue, green respectively).

#### Empirical benchmarks (parity tokens)

Derived from 500 random 5-player games (blue wires 1-12, all choices per player, n=24,000). Mean uncertainty resolved per parity indication is approximately 5.0%, with a standard deviation of approximately 1.5%.

| Range | Interpretation | Statistical context |
|-------|---------------|---------------------|
| < 3.5% | Poor. The parity constraint barely narrows possibilities. | Below mean - 1 sigma |
| 3.5% - 5% | Moderate. Useful partitioning for teammates. | Between mean - 1 sigma and mean |
| 5% - 7% | Good. Noticeably reduces stand uncertainty. | Between mean and mean + 1 sigma |
| >= 7% | Excellent. Strong partitioning effect. | Above mean + 1 sigma |

#### Empirical benchmarks (multiplicity tokens)

Multiplicity reveals only xM count (not value), so IG is lower than standard or parity. With true joint entropy, all IG values are non-negative. Thresholds need recalibration — these are conservative placeholders.

| Range | Interpretation |
|-------|---------------|
| < 1% | Poor. Minimal constraint on the stand. |
| 1% - 3.5% | Moderate. Some useful information for teammates. |
| 3.5% - 5% | Good. Noticeably constrains possibilities. |
| >= 5% | Excellent. Strong multiplicity constraint. |

#### When to use which metric

- **Bits** are meaningful in absolute terms: 3 bits always means an 8x reduction in possibilities, regardless of stand size. Use bits to understand the raw power of an indication.
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

- **Generic observer model**: The metric uses the full complement pool (all wires in play minus publicly known wires from other stands). Individual observers have smaller pools (they also know their own wires), which means the actual entropy reduction they experience may differ slightly. The relative ranking of indication choices is robust to this.
- **Uncertain (X-of-Y) wire groups**: Handled by including all candidates in the pool, which slightly overestimates uncertainty. The ranking remains valid.
