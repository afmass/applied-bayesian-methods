# Applied Bayesian Methods — Short Course Materials

This repository centralizes slides, exercises, and supporting scripts for the Applied Bayesian Methods short course.

## Course information

| | |
|---|---|
| **Location** | AgroParisTech, 14 rue Girardet, Nancy |
| **Dates** | 7–9 April 2026 |
| **Registration deadline** | 30 March 2026 |
| **Keywords** | Statistics, Modelling, Spatio-temporal, Mixed-effects, Ensemble approaches |

## Objectives

By the end of this course, participants will be able to:

- Understand the Bayesian framework and its advantages
- Specify, fit, and interpret hierarchical and mixed Bayesian models
- Incorporate spatial and temporal dependencies into models
- Evaluate, compare, and validate Bayesian models
- Understand and apply ensemble modelling approaches
- Apply Bayesian tools to their own research data
- Report results transparently and reproducibly

## Programme

### Day 1 — Bayesian Foundations and Hierarchical Models
- Theoretical background and introduction to the Bayesian framework
- Fitting and interpreting Bayesian mixed effect models

**Software:** R (`rstanarm`, `brms`)  
**Instructor:** Alexander MASSEY → [`/Day1`](./Day1)

### Day 2 — Space and Time in the Bayesian Framework
- Space and time modelling concepts and approaches in the Bayesian framework
- Fitting and evaluating models with explicit space and/or time components

**Software:** R (INLA)  
**Instructor:** Lionel HERTZOG → [`/Day2`](./Day2)

### Day 3 — Ensemble Modelling and Bayesian Model Averaging
- Introduction to ensemble modelling methods with a focus on Bayesian Model Averaging
- Applying and evaluating Bayesian Model Averaging on real-world models

**Software:** Python  
**Instructor:** Nikola BESIC → [`/Day3`](./Day3)

## Prerequisites

- Basic statistical concepts (means, correlation, distributions, probability) and tools (regression, correlation)
- Working knowledge of at least one programming language (R, Python, or equivalent)

## Teaching methods

- Short presentations covering major conceptual knowledge
- Simulated and real-world case studies
- Live coding sessions (R and Python)
- Hands-on practical work in small groups

## Repository structure

```
/day1/      # Day 1 slides, scripts, and exercises (R)
/day2/      # Day 2 slides, scripts, and exercises (R / INLA)
/day3/      # Day 3 slides, scripts, and exercises (Python)
README.md
```
