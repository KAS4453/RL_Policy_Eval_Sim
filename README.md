# Evaluating Contextual Bandit-Based Recommendation for Mental Health Interventions

---

## Overview

Digital mental health interventions provide a scalable way to support well-being, especially in resource-constrained settings. However, their effectiveness depends strongly on context, as user responses vary across situations and individuals.

This work models intervention delivery as an **interactive recommendation problem**, where recommendations are selected based on user context and updated using observed feedback over time.

---

## Objective

To evaluate how **adaptive decision-making methods** perform in real-world mental health settings and compare their behavior against simulated environments.

---

## Approach

- Developed a system for **context-aware intervention delivery**
- Modeled user interaction as a **sequential decision-making process**
- Incorporated an **adaptive learning strategy** that updates recommendations over time
- Compared against a **non-adaptive baseline**
- Evaluated across:
  - **Simulated environment**
  - **Real-world deployment**

---

## Real-World Study

- Conducted a **2-week in-the-wild study**
- Involved **university students**
- Delivered interventions through a mobile-based platform
- Collected **implicit feedback** based on user engagement

---

## Evaluation

The study evaluates:

- **Overall performance** (mean reward)
- **Context-specific effectiveness** (activity-wise analysis)
- **Temporal trends** (changes over time)
- **Exploration vs exploitation behavior**
- **Learning dynamics** (uncertainty reduction and decision confidence)

---

## Key Findings

- **Simulation**
  - Adaptive approach: **0.79**
  - Baseline: **0.53**

- **Real-world deployment**
  - Adaptive approach: **0.62**

- A clear **sim-to-real gap** is observed, highlighting challenges such as:
  - Noisy and delayed human feedback  
  - Contextual variability  
  - Behavioral unpredictability  

---

## Note

An anonymized version of the implementation is provided for reproducibility.

---

## Repository Structure & File Descriptions

### 🔹 Core Simulation Modules

* **`environment.py`**
  Defines the JITAI environment:

  * User activities and intervention space
  * Context-specific feasibility constraints
  * Engagement probability modeling
  * Stochastic reward generation using Dirichlet distributions
  * `JITAIEnvironment` class simulates user responses to interventions

* **`policy.py`**
  Implements decision-making strategies:

  * `RandomPolicy` — non-adaptive baseline
  * `ThompsonDirichletPolicy` — adaptive contextual bandit using Dirichlet-Multinomial modeling
  * Handles action selection and online learning (posterior updates)

* **`simulator.py`**
  Runs the full simulation:

  * Simulates multi-day participant interactions
  * Integrates environment + policy
  * Generates sequential logged data (context, action, reward)
  * Supports missing notifications and realistic scheduling

* **`utils.py`**
  Helper utilities:

  * Logging setup
  * Time and timestamp generation
  * Participant ID creation
  * Missing data simulation
  * Summary statistics printing

---

### 🔹 Data Generation

* **`generate_dataset.py`**
  CLI and programmatic interface to generate datasets:

  * Supports Random, Thompson Sampling, or both
  * Configurable participants, duration, missing rate
  * Outputs ready-to-use CSV datasets

---

### 🔹 Analysis & Visualization

* **`plot.ipynb`**
  End-to-end analysis notebook:

  * Performance comparison (Random vs TS, Sim vs Real)
  * Exploration vs exploitation dynamics (entropy-based)
  * Posterior learning behavior (uncertainty + confidence)
  * Activity-wise reward analysis
  * Bootstrap confidence intervals
  * Generates publication-ready plots and tables

---

### 🔹 Outputs

* **`outputs/`** *(generated artifacts)*

  * `activity_comparison_table.csv`
    Mean reward per activity across policies

  * `summary_comparison_table.csv`
    Overall dataset statistics and reward distribution

  * `behavior_analysis_table.csv`
    Exploration–exploitation metrics (entropy trends)

  * `activity_random_vs_ts_sim.png`
    Activity-wise comparison (Random vs TS in simulation)

  * `plot_sim_random_vs_ts.pdf`
    User-level reward comparison (simulation)

  * `plot_ts_sim_vs_real.pdf`
    Sim-to-real comparison for Thompson Sampling

---

### 🔹 Miscellaneous

* **`LICENSE`**
  License information for usage and distribution

* **`README.md`**
  Project overview, methodology, and usage instructions
