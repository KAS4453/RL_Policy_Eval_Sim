import datetime as _dt
import logging
from typing import List, Dict

import numpy as np
import pandas as pd

from environment import JITAIEnvironment, INTERVENTION_NAMES
from policy import BasePolicy, create_policy
from utils import (
    build_timestamp,
    generate_participant_ids,
    get_day_name,
    get_day_time,
    hours_in_day,
    make_study_start,
    setup_logger,
    should_miss_notification,
)

logger = setup_logger("simulator")

# Final column order — Context added between day_time and Intervention
OUTPUT_COLUMNS = [
    "Participant_Id",
    "NT_post_Timestamp",
    "day",
    "day_time",
    "Context",
    "Intervention",
    "Status",
    "Reward",
]

# Per-participant simulation

def simulate_participant(participant_id: str, n_days: int, policy: BasePolicy, miss_rate: float = 0.25, base_seed: int = 42,) -> List[Dict]:
    env = JITAIEnvironment(participant_id=participant_id, base_seed=base_seed)

    # Separate RNG for missing-slot draws (independent of env and policy)
    numeric_id = int(participant_id.lstrip("P"))
    slot_rng = np.random.default_rng(base_seed + numeric_id * 13 + 99)

    study_start = make_study_start()
    rows: List[Dict] = []

    for study_day in range(n_days):
        current_dt = study_start + _dt.timedelta(days=study_day)
        day_name   = get_day_name(current_dt)

        for hour in hours_in_day():
            timestamp = build_timestamp(study_start, study_day, hour)
            day_time  = get_day_time(hour)

            # Truly missing slot — no notification delivered
            if should_miss_notification(slot_rng, miss_rate):
                rows.append({
                    "Participant_Id":    participant_id,
                    "NT_post_Timestamp": timestamp,
                    "day":               day_name,
                    "day_time":          day_time,
                    "Context":           None,
                    "Intervention":      None,
                    "Status":            None,
                    "Reward":            None,
                })
                continue

            # Sample current activity (observable context)
            activity = env.sample_activity()

            # Policy selects an intervention (returns letter code)
            intervention_code, _ = policy.select_action(context=activity)

            # Resolve full intervention name
            intervention_name = INTERVENTION_NAMES[intervention_code]

            # Environment returns participant response
            outcome = env.step(
                activity=activity,
                intervention_code=intervention_code,
            )

            # Policy learns from the observed reward
            policy.update(
                context=activity,
                action=intervention_code,
                reward=outcome["reward"],
            )

            rows.append({
                "Participant_Id":    participant_id,
                "NT_post_Timestamp": timestamp,
                "day":               day_name,
                "day_time":          day_time,
                "Context":           activity,
                "Intervention":      intervention_name,
                "Status":            outcome["status"],
                "Reward":            outcome["reward"],
            })

    return rows


# Full simulation across all participants

def run_simulation(n_participants: int = 100, n_days_range: tuple = (10, 14), policy_name: str = "thompson", miss_rate: float = 0.25, base_seed: int = 42, verbose: bool = True,) -> pd.DataFrame:
    global_rng      = np.random.default_rng(base_seed)
    participant_ids = generate_participant_ids(n_participants)
    all_rows: List[Dict] = []

    if verbose:
        logger.info("=" * 60)
        logger.info("  JITAI Simulation")
        logger.info("=" * 60)
        logger.info(f"  Policy        : {policy_name}")
        logger.info(f"  Participants  : {n_participants}")
        logger.info(f"  Days range    : {n_days_range[0]}–{n_days_range[1]}")
        logger.info(f"  Miss rate     : {miss_rate:.0%}")
        logger.info(f"  Seed          : {base_seed}")
        logger.info("=" * 60)

    for i, pid in enumerate(participant_ids):
        n_days = int(global_rng.integers(n_days_range[0], n_days_range[1] + 1))

        # Each participant gets their own fresh policy instance so they do not share knowledge (independent bandit per participant).
        numeric_pid      = int(pid.lstrip("P"))
        participant_seed = base_seed + numeric_pid
        p_policy         = create_policy(policy_name=policy_name, seed=participant_seed)

        rows = simulate_participant(
            participant_id=pid,
            n_days=n_days,
            policy=p_policy,
            miss_rate=miss_rate,
            base_seed=base_seed,
        )
        all_rows.extend(rows)

        if verbose and (i + 1) % 10 == 0:
            logger.info(f"  Simulated {i + 1}/{n_participants} participants …")

    if verbose:
        logger.info(f"  Complete. Total rows: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]
    return df