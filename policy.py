import numpy as np
from typing import Dict, List, Tuple

from environment import (
    ACTIVITIES,
    INTERVENTION_CODES,
    FEASIBLE_INTERVENTIONS,
    RESPONSE_DISTRIBUTIONS,
)

# How strongly the expert priors are weighted
PRIOR_CONCENTRATION: float = 3.0

# Base interface

class BasePolicy:
    def select_action(self, context: str) -> Tuple[str, Dict]:
        raise NotImplementedError

    def update(self, context: str, action: str, reward: float) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

# Random Policy (baseline)

class RandomPolicy(BasePolicy):
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._counts: Dict[str, int] = {c: 0 for c in INTERVENTION_CODES}

    @property
    def name(self) -> str:
        return "Random"

    def select_action(self, context: str) -> Tuple[str, Dict]:
        chosen = str(self.rng.choice(INTERVENTION_CODES))
        self._counts[chosen] += 1
        info = {
            "policy":     "Random",
            "context":    context,
            "propensity": 1.0 / len(INTERVENTION_CODES),
            "n_arms":     len(INTERVENTION_CODES),
        }
        return chosen, info

    def update(self, context: str, action: str, reward: float) -> None:
        pass

    def get_counts(self) -> Dict[str, int]:
        return dict(self._counts)

# Thompson Sampling — Dirichlet-Multinomial, one bandit per context

class ThompsonDirichletPolicy(BasePolicy):

    def __init__(self, seed: int = 42, concentration: float = PRIOR_CONCENTRATION):
        self.rng = np.random.default_rng(seed)
        self.concentration = concentration

        # Build per-context Dirichlet parameter arrays
        # Only feasible arms are included for each activity
        self.alpha: Dict[str, Dict[str, np.ndarray]] = {}

        for activity in ACTIVITIES:
            self.alpha[activity] = {}
            for code in FEASIBLE_INTERVENTIONS[activity]:
                p_yes, p_partial, p_no = RESPONSE_DISTRIBUTIONS[(activity, code)]
                prior = concentration * np.array(
                    [max(p_yes, 0.01), max(p_partial, 0.01), max(p_no, 0.01)],
                    dtype=float,
                )
                self.alpha[activity][code] = prior

    @property
    def name(self) -> str:
        return "ThompsonDirichlet"

    def select_action(self, context: str) -> Tuple[str, Dict]:
        arms  = self.alpha[context]  # {code: α_array}
        codes = list(arms.keys())

        # Sample expected reward for every arm simultaneously
        expected_rewards = np.zeros(len(codes))
        for i, code in enumerate(codes):
            theta = self.rng.dirichlet(arms[code])      # [p_yes, p_partial, p_no]
            expected_rewards[i] = 1.0 * theta[0] + 0.5 * theta[1]

        chosen_idx = int(np.argmax(expected_rewards))
        chosen     = codes[chosen_idx]

        alpha_chosen   = arms[chosen]
        posterior_mean = alpha_chosen / alpha_chosen.sum()

        info = {
            "policy":              "ThompsonDirichlet",
            "context":             context,
            "n_feasible_arms":     len(codes),
            "chosen_E_reward":     float(expected_rewards[chosen_idx]),
            "posterior_mean_yes":  float(posterior_mean[0]),
            "posterior_mean_part": float(posterior_mean[1]),
            "posterior_mean_no":   float(posterior_mean[2]),
        }
        return chosen, info

    # ------------------------------------------------------------------

    def update(self, context: str, action: str, reward: float) -> None:
        if context not in self.alpha:
            return
        if action not in self.alpha[context]:
            return  # infeasible arm was somehow chosen — silently ignore

        if reward == 1.0:
            self.alpha[context][action][0] += 1.0
        elif reward == 0.5:
            self.alpha[context][action][1] += 1.0
        else:
            self.alpha[context][action][2] += 1.0

    def get_arm_stats(self, context: str) -> Dict:
        if context not in self.alpha:
            return {}
        result = {}
        for code, alpha in self.alpha[context].items():
            total = alpha.sum()
            result[code] = {
                "alpha":          alpha.tolist(),
                "posterior_mean": (alpha / total).tolist(),  # [p_yes, p_part, p_no]
                "E_reward":       float(1.0 * (alpha[0] / total) + 0.5 * (alpha[1] / total)),
                "n_updates":      int(total - self.concentration * 3),  # approx
            }
        return result

# Factory

def create_policy(policy_name: str = "thompson", seed: int = 42) -> BasePolicy:
    name = policy_name.lower().strip()

    if name in ("thompson", "ts"):
        return ThompsonDirichletPolicy(seed=seed)

    if name in ("random", "rand"):
        return RandomPolicy(seed=seed)

    raise ValueError(
        f"Unknown policy '{policy_name}'. Choose 'thompson' or 'random'."
    )