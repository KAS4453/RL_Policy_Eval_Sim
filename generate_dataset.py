import argparse
import os
import sys

import pandas as pd

from simulator import run_simulation
from utils import print_summary, set_global_seed, setup_logger

logger = setup_logger("generate_dataset")

# Default output file names
DEFAULT_RANDOM_OUTPUT = "MasterEmotionsWithMood_Simulated_Random.csv"
DEFAULT_TS_OUTPUT     = "MasterEmotionsWithMood_Simulated_TS.csv"


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic JITAI datasets.\n"
            "By default produces BOTH a Random-policy CSV and a "
            "Thompson-Sampling CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--participants", type=int,   default=100)
    parser.add_argument("--days-min",    type=int,   default=10)
    parser.add_argument("--days-max",    type=int,   default=14)
    parser.add_argument("--miss-rate",   type=float, default=0.25)
    parser.add_argument(
        "--policy", type=str, default="both",
        choices=["both", "thompson", "random"],
        help="Which policy to run. 'both' produces two CSVs.",
    )
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output path (only used when --policy is 'thompson' or 'random'). "
            "Ignored when --policy=both."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


# Core runner

def _run_and_save(policy_name: str, output_path: str, n_participants: int, days_min: int, days_max: int, miss_rate: float, seed: int, verbose: bool,) -> pd.DataFrame:
    
    df = run_simulation(n_participants=n_participants, n_days_range=(days_min, days_max), policy_name=policy_name, miss_rate=miss_rate, base_seed=seed, verbose=verbose,)

    if verbose:
        print_summary(df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {os.path.abspath(output_path)}")
    logger.info(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

# Main (CLI)

def main() -> None:
    args = parse_args()

    # Validate
    if args.participants < 1:
        logger.error("--participants must be >= 1.")
        sys.exit(1)
    if not 0.0 <= args.miss_rate <= 1.0:
        logger.error("--miss-rate must be in [0, 1].")
        sys.exit(1)
    if args.days_min > args.days_max:
        logger.error("--days-min must be <= --days-max.")
        sys.exit(1)

    set_global_seed(args.seed)
    verbose = not args.quiet

    common_kwargs = dict(
        n_participants=args.participants,
        days_min=args.days_min,
        days_max=args.days_max,
        miss_rate=args.miss_rate,
        seed=args.seed,
        verbose=verbose,
    )

    if args.policy == "both":
        logger.info("Running Random policy …")
        _run_and_save(policy_name="random",   output_path=DEFAULT_RANDOM_OUTPUT, **common_kwargs)

        logger.info("Running Thompson Sampling policy …")
        _run_and_save(policy_name="thompson", output_path=DEFAULT_TS_OUTPUT,     **common_kwargs)

    else:
        # Single policy run
        if args.output is None:
            output_path = (
                DEFAULT_TS_OUTPUT if args.policy == "thompson"
                else DEFAULT_RANDOM_OUTPUT
            )
        else:
            output_path = args.output

        _run_and_save(policy_name=args.policy, output_path=output_path, **common_kwargs)


# Programmatic API (import and call directly)

def generate_dataset(n_participants: int = 100, days_min: int = 10, days_max: int = 14, miss_rate: float = 0.25, policy: str = "both", seed: int = 42, output_path_random: str = DEFAULT_RANDOM_OUTPUT, output_path_ts: str = DEFAULT_TS_OUTPUT, verbose: bool = True,) -> dict:

    set_global_seed(seed)
    results = {}

    common_kwargs = dict(
        n_participants=n_participants,
        days_min=days_min,
        days_max=days_max,
        miss_rate=miss_rate,
        seed=seed,
        verbose=verbose,
    )

    if policy in ("both", "random"):
        if verbose:
            logger.info("Running Random policy …")
        df_random = _run_and_save(
            policy_name="random",
            output_path=output_path_random,
            **common_kwargs,
        )
        results["random"] = df_random

    if policy in ("both", "thompson"):
        if verbose:
            logger.info("Running Thompson Sampling policy …")
        df_ts = _run_and_save(
            policy_name="thompson",
            output_path=output_path_ts,
            **common_kwargs,
        )
        results["thompson"] = df_ts

    return results


if __name__ == "__main__":
    main()