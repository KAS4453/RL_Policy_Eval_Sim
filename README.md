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
