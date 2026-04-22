import numpy as np
from typing import Dict, List, Tuple

# Intervention definitions (letter code → full name)

INTERVENTION_NAMES: Dict[str, str] = {
    "A": "Breathing Exercise",
    "B": "Calling or texting loved ones",
    "C": "Close your eyes and try to hear ambient sounds",
    "D": "Doing Simple Neck Rolls to Release Tension",
    "E": "Eating something you like",
    "F": "Go for a small walk",
    "G": "Journal Writing",
    "H": "Listening to Music",
    "I": "Make a list of positive things inside you",
    "J": "Observe your surroundings",
    "K": "Play game",
    "L": "Scroll through old memories from your gallery",
    "M": "Stretching",
    "N": "Watching funny videos",
    "O": "Watching motivational video",
    "P": "Writing Down a Worry and Putting It Aside",
}

INTERVENTION_CODES: List[str] = list(INTERVENTION_NAMES.keys())  # ["A", … "P"]


# Activity definitions

ACTIVITIES: List[str] = [
    "Attending Lecture",
    "Exercise",
    "Relaxing",
    "E-Rick/Auto",
    "Cycling",
    "Walking",
    "Running",
    "Studying",
    "Eating",
    "Standing",
]

# This we got from a previous study
# Activity sampling weights (realistic daily frequency)
ACTIVITY_WEIGHTS: List[float] = [
    0.12,   # Attending Lecture
    0.08,   # Exercise
    0.15,   # Relaxing
    0.10,   # E-Rick/Auto
    0.05,   # Cycling
    0.12,   # Walking
    0.05,   # Running
    0.18,   # Studying
    0.10,   # Eating
    0.05,   # Standing
]
_w = np.array(ACTIVITY_WEIGHTS, dtype=float)
ACTIVITY_WEIGHTS_NORM: np.ndarray = _w / _w.sum()


# Rows: interventions A–P | Columns: activities (same order as ACTIVITIES)
# Column order: Lec   Ex    Rel   Veh   Cyc   Walk  Run   Study Eat   Stand
#
# For non-feasible (activity, intervention) pairs the raw table valuev is still stored (used in infeasible arm partial-engagement calc); feasibility is enforced separately via FEASIBLE_INTERVENTIONS.

_RAW_TABLE: Dict[str, List[float]] = {
    #        Lec   Ex    Rel   Veh   Cyc   Walk  Run   Study Eat   Stand
    "A": [0.8,  0.9,  0.9,  0.7,  0.4,  0.9,  0.5,  0.9,  0.2,  0.9],
    "B": [0.2,  0.5,  0.9,  0.9,  0.4,  0.9,  0.3,  0.6,  0.9,  0.9],
    "C": [0.5,  0.9,  0.9,  0.9,  0.4,  0.8,  0.7,  0.9,  0.5,  0.9],
    "D": [0.9,  0.9,  0.9,  0.9,  0.3,  0.9,  0.5,  0.9,  0.5,  0.9],
    "E": [0.2,  0.0,  0.9,  0.5,  0.0,  0.5,  0.0,  0.8,  1.0,  0.7],
    "F": [0.1,  0.6,  0.7,  0.0,  0.0,  0.8,  0.4,  0.9,  0.1,  0.8],
    "G": [0.4,  0.0,  0.8,  0.2,  0.0,  0.1,  0.1,  0.7,  0.1,  0.4],
    "H": [0.2,  0.9,  0.9,  0.9,  0.8,  0.9,  0.9,  0.7,  0.8,  0.9],
    "I": [0.7,  0.4,  0.9,  0.6,  0.2,  0.4,  0.2,  0.7,  0.4,  0.7],
    "J": [0.8,  0.9,  0.9,  0.9,  0.7,  0.9,  0.7,  0.8,  0.9,  0.9],
    "K": [0.3,  0.1,  0.8,  0.6,  0.0,  0.3,  0.1,  0.4,  0.3,  0.8],
    "L": [0.5,  0.3,  0.9,  0.9,  0.2,  0.7,  0.2,  0.4,  0.6,  0.9],
    "M": [0.4,  0.9,  0.9,  0.2,  0.1,  0.7,  0.5,  0.9,  0.1,  0.9],
    "N": [0.2,  0.1,  0.9,  0.8,  0.2,  0.8,  0.0,  0.4,  0.9,  0.9],
    "O": [0.2,  0.7,  0.9,  0.8,  0.2,  0.7,  0.1,  0.6,  0.6,  0.9],
    "P": [0.6,  0.2,  0.8,  0.4,  0.0,  0.2,  0.0,  0.7,  0.2,  0.5],
}

# Nested dict: PROB_MATRIX[intervention_code][activity] = p_engage
PROB_MATRIX: Dict[str, Dict[str, float]] = {
    code: {act: prob for act, prob in zip(ACTIVITIES, probs)}
    for code, probs in _RAW_TABLE.items()
}


# Feasibility constraints
# Only these interventions are contextually valid for each activity.
# Probabilities sourced from activity_interventions domain-expert dict.

FEASIBLE_INTERVENTIONS: Dict[str, List[str]] = {
    "Attending Lecture": ["A", "C", "D", "G", "I", "J", "L", "P"],
    "Exercise":          ["A", "B", "C", "D", "F", "H", "I", "J", "M", "O"],
    "Relaxing":          ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                          "K", "L", "M", "N", "O", "P"],
    "E-Rick/Auto":         ["A", "B", "C", "D", "E", "H", "I", "J", "K", "L",
                          "N", "O", "P"],
    "Cycling":           ["C", "H", "J"],
    "Walking":           ["A", "B", "C", "D", "E", "F", "H", "I", "J", "L",
                          "M", "N", "O"],
    "Running":           ["A", "C", "D", "H", "J", "M"],
    "Studying":          ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                          "K", "L", "M", "O", "P"],
    "Eating":            ["B", "C", "D", "E", "H", "J", "L", "N", "O"],
    "Standing":          ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K",
                          "L", "M", "N", "O", "P"],
}

# Engagement probabilities per feasible (activity, intervention)
# These override the raw table for feasible arms.

_FEASIBLE_PROBS: Dict[str, Dict[str, float]] = {
    "Attending Lecture": {"A": 0.8, "C": 0.5, "D": 0.9, "G": 0.4,
                          "I": 0.7, "J": 0.8, "L": 0.5, "P": 0.6},
    "Exercise":          {"A": 0.9, "B": 0.5, "C": 0.9, "D": 0.9,
                          "F": 0.6, "H": 0.9, "I": 0.4, "J": 0.9,
                          "M": 0.9, "O": 0.7},
    "Relaxing":          {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.9,
                          "E": 0.9, "F": 0.7, "G": 0.8, "H": 0.9,
                          "I": 0.9, "J": 0.9, "K": 0.8, "L": 0.9,
                          "M": 0.9, "N": 0.9, "O": 0.9, "P": 0.8},
    "E-Rick/Auto":         {"A": 0.7, "B": 0.9, "C": 0.9, "D": 0.9,
                          "E": 0.5, "H": 0.9, "I": 0.6, "J": 0.9,
                          "K": 0.6, "L": 0.9, "N": 0.8, "O": 0.8,
                          "P": 0.4},
    "Cycling":           {"C": 0.4, "H": 0.8, "J": 0.7},
    "Walking":           {"A": 0.9, "B": 0.9, "C": 0.8, "D": 0.9,
                          "E": 0.5, "F": 0.8, "H": 0.9, "I": 0.4,
                          "J": 0.9, "L": 0.7, "M": 0.7, "N": 0.8,
                          "O": 0.7},
    "Running":           {"A": 0.5, "C": 0.7, "D": 0.5, "H": 0.9,
                          "J": 0.7, "M": 0.5},
    "Studying":          {"A": 0.9, "B": 0.6, "C": 0.9, "D": 0.9,
                          "E": 0.8, "F": 0.9, "G": 0.7, "H": 0.7,
                          "I": 0.7, "J": 0.8, "K": 0.4, "L": 0.4,
                          "M": 0.9, "O": 0.6, "P": 0.7},
    "Eating":            {"B": 0.9, "C": 0.5, "D": 0.5, "E": 1.0,
                          "H": 0.8, "J": 0.9, "L": 0.6, "N": 0.9,
                          "O": 0.6},
    "Standing":          {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.9,
                          "E": 0.7, "F": 0.8, "H": 0.9, "I": 0.7,
                          "J": 0.9, "K": 0.8, "L": 0.9, "M": 0.9,
                          "N": 0.9, "O": 0.9, "P": 0.5},
}


# Response distribution builder

def _build_response_distribution(activity: str, code: str, rng: np.random.Generator,) -> Tuple[float, float, float]:

    feasible = code in FEASIBLE_INTERVENTIONS[activity]

    if feasible:
        p = _FEASIBLE_PROBS[activity][code]
        alpha = np.array([
            0.85 * p,
            0.15 * p,
            1.0 - p,
        ])
    else:
        p = PROB_MATRIX[code][activity]
        alpha = np.array([
            0.0,
            0.70 * p,
            1.0 - 0.70 * p,
        ])

    # Scale to control variance; add small floor to avoid degenerate Dirichlet
    concentration = 20
    alpha = alpha * concentration + 1e-3

    probs = rng.dirichlet(alpha)
    return tuple(float(x) for x in probs)


# Module-level RNG used exclusively to pre-compute the stationary
# response distributions.  Fixed seed guarantees reproducibility
# across imports without coupling to participant or policy seeds.
_DIST_RNG: np.random.Generator = np.random.default_rng(seed=0)

# Pre-compute all distributions once (stationary environment).
# RESPONSE_DISTRIBUTIONS[(activity, code)] = (p_yes, p_partial, p_no)
RESPONSE_DISTRIBUTIONS: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    (activity, code): _build_response_distribution(activity, code, _DIST_RNG)
    for activity in ACTIVITIES
    for code in INTERVENTION_CODES
}


# ---------------------------------------------------------------------------
# Participant environment class
# ---------------------------------------------------------------------------

class JITAIEnvironment:

    STATUS_OUTCOMES = ("yes", "yes_but_not_feasible", "no")
    REWARDS         = (1.0,   0.5,                    0.0)

    def __init__(self, participant_id: str, base_seed: int = 42):
        numeric = int(participant_id.lstrip("P"))
        self.participant_id = participant_id
        self.rng = np.random.default_rng(base_seed + numeric * 7)
        self.response_distributions = {
            (activity, code): _build_response_distribution(activity, code, self.rng)
            for activity in ACTIVITIES
            for code in INTERVENTION_CODES
        }
        
    # Core interaction

    def step(self, activity: str, intervention_code: str) -> Dict:
        base_dist = self.response_distributions[(activity, intervention_code)]

        # Add Gaussian noise
        noise = self.rng.normal(0, 0.05, size=3)   # sigma = 0.05 (tunable)

        noisy = np.array(base_dist) + noise

        # Avoid negative probabilities
        noisy = np.clip(noisy, 1e-3, None)

        # Normalize to sum to 1
        noisy = noisy / noisy.sum()

        outcome_idx = int(self.rng.choice(3, p=noisy))

        return {
            "status":            self.STATUS_OUTCOMES[outcome_idx],
            "reward":            self.REWARDS[outcome_idx],
            "intervention_name": INTERVENTION_NAMES[intervention_code],
        }

    # Activity sampling

    def sample_activity(self) -> str:
        """Sample a plausible current activity for this notification slot."""
        return str(self.rng.choice(ACTIVITIES, p=ACTIVITY_WEIGHTS_NORM))