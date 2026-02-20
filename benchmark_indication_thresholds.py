"""Benchmark indication quality across 500 random games for all 3 types.

Generates random 5-player games and computes the best indication for
each player using standard, parity, and multiplicity analysis. Collects
uncertainty_resolved statistics (mean, std) for each type to calibrate
color-coded thresholds.

Standard and parity use blue wires 1-12 with 0-4 yellow, 0-3 red.
Multiplicity uses blue wires only (1-12) — multiplicity tokens can work
with yellow but the benchmarks use blue-only for simplicity and to match
the typical mission profile.
"""

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bomb_busters
import compute_probabilities


def main() -> None:
    num_games = 500

    # Collect best-indication uncertainty_resolved for each player
    standard_pcts: list[float] = []
    parity_pcts: list[float] = []
    multiplicity_pcts: list[float] = []

    for game_idx in range(num_games):
        seed = game_idx * 7 + 1

        # Standard + parity: use mixed wire pool (0-4 yellow, 0-3 red)
        # Use deterministic yellow/red counts from seed
        import random
        rng = random.Random(seed)
        n_yellow = rng.randint(0, 4)
        n_red = rng.randint(0, 3)

        game_std = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D", "E"],
            seed=seed,
            yellow_wires=n_yellow if n_yellow > 0 else None,
            red_wires=n_red if n_red > 0 else None,
        )

        # Multiplicity: blue only
        game_mult = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D", "E"],
            seed=seed,
        )

        for player_idx in range(5):
            # Standard indication
            std_choices = compute_probabilities.rank_indications(
                game_std, player_idx,
            )
            if std_choices:
                standard_pcts.append(std_choices[0].uncertainty_resolved)

            # Parity indication (no yellow wires allowed in parity missions,
            # but we still compute from the same game — parity only considers
            # blue wires anyway)
            parity_choices = compute_probabilities.rank_indications_parity(
                game_std, player_idx,
            )
            if parity_choices:
                parity_pcts.append(parity_choices[0].uncertainty_resolved)

            # Multiplicity indication
            mult_choices = compute_probabilities.rank_indications_multiplicity(
                game_mult, player_idx,
            )
            if mult_choices:
                multiplicity_pcts.append(mult_choices[0].uncertainty_resolved)

        if (game_idx + 1) % 50 == 0:
            print(f"  Completed {game_idx + 1}/{num_games} games...")

    print()
    print("=" * 60)
    print("INDICATION TYPE BENCHMARKS (500 games, 5 players each)")
    print("=" * 60)

    for name, pcts in [
        ("Standard Info Token", standard_pcts),
        ("Even/Odd Parity", parity_pcts),
        ("x1/x2/x3 Multiplicity", multiplicity_pcts),
    ]:
        if not pcts:
            print(f"\n{name}: No data")
            continue

        mean = statistics.mean(pcts)
        stdev = statistics.stdev(pcts) if len(pcts) > 1 else 0.0
        q25 = sorted(pcts)[len(pcts) // 4]
        median = statistics.median(pcts)
        q75 = sorted(pcts)[3 * len(pcts) // 4]
        min_val = min(pcts)
        max_val = max(pcts)

        print(f"\n{name} (n={len(pcts)}):")
        print(f"  Mean:   {mean:.1%}")
        print(f"  StdDev: {stdev:.1%}")
        print(f"  Min:    {min_val:.1%}")
        print(f"  Q25:    {q25:.1%}")
        print(f"  Median: {median:.1%}")
        print(f"  Q75:    {q75:.1%}")
        print(f"  Max:    {max_val:.1%}")
        print()
        print(f"  Suggested thresholds:")
        print(f"    Green (excellent): >= {mean + stdev:.1%}  (mean + 1σ)")
        print(f"    Blue  (good):      >= {mean:.1%}  (mean)")
        print(f"    Yellow (moderate):  >= {mean - stdev:.1%}  (mean - 1σ)")
        print(f"    Red   (poor):      <  {mean - stdev:.1%}")


if __name__ == "__main__":
    main()
