# Day 1 — Bayesian Foundations and Hierarchical Models in R

This folder contains all materials for Day 1 of the Applied Bayesian Methods course

## Topics

1.  **The Bayesian framework** — Bayes' theorem, posterior inference, the likelihood principle and its connection to Bayesian updating
2.  **Choice of priors** — weakly informative, conjugate, and regularising priors; prior predictive checks
3.  **Mixed effects models as Bayesian hierarchical models** — random effects with different covariance structures (independent, correlated, spatial)
4.  **Connections to penalized regression** — ridge/lasso as Bayesian shrinkage; equivalence between regularisation penalties and prior distributions
5.  **Temporal random effects in INLA** *(time permitting)* — a brief applied example to bridge into Day 2

## Software

-   **Language:** R
-   **Primary packages:** `rstanarm`, `brms`
-   **Supporting packages:** `bayesplot`, `loo`, `tidybayes`, `lme4` (for reference/comparison)

## Folder structure

```         
day1/
├── exercises/
├── slides/
```

## Notes

-   Slides are built with Reveal.js via Quarto (`format: revealjs`).
-   Rendered output (e.g. `*.html`, `*_files/`) is gitignored at the repo level — do not force-add these.
-   Stan models may take time to compile on first run; pre-compiled model objects (`.rds`) can be saved to `scripts/` to speed up live demos.
