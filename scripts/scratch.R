# ── Simulation (run once; reuse in Part 2) ─────────────────────────────
library(MASS)
library(brms)
library(tidyverse)

set.seed(42)
n <- 200    # observations
p <- 15     # predictors — all non-zero coefficients

# Block correlation: three groups of 5 correlated predictors (rho = 0.7)
# This ensures GCV finds a genuine interior minimum
blk <- function(rho, k) { M <- matrix(rho, k, k); diag(M) <- 1; M }
Sigma <- as.matrix(Matrix::bdiag(blk(0.7,5), blk(0.7,5), blk(0.7,5)))
X_sim <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
X_sim <- scale(X_sim)   # standardise columns to unit variance

# True coefficients: all non-zero, decreasing in magnitude across blocks
beta_true <- c( 1.5,  1.2, -1.0,  0.8, -0.6,    # block 1 — largest
                1.0, -0.9,  0.7, -0.5,  0.4,    # block 2 — medium
                0.8,  0.6, -0.5,  0.3, -0.2)    # block 3 — smallest
sigma_true <- 1.0

y_sim <- as.numeric(X_sim %*% beta_true + rnorm(n, sd = sigma_true))
y_sim <- scale(y_sim)   # standardise response

df_sim <- as.data.frame(cbind(y = as.numeric(y_sim), X_sim))
colnames(df_sim) <- c('y', paste0('x', 1:p))


# GCV grid: start fine over a range you expect the minimum to fall in
# With scaled predictors and scaled response, optimal lambda is typically 0.5–5
ridge_grid <- lm.ridge(y ~ ., data = df_sim, lambda = seq(0, 4, by = 0.01))

# ALWAYS plot before accepting the minimum
plot(ridge_grid$lambda, ridge_grid$GCV, type = 'l',
     xlab = expression(lambda), ylab = 'GCV',
     main = 'GCV curve — check for interior minimum')
abline(v = as.numeric(names(which.min(ridge_grid$GCV))), col='red', lty=2)



lambda_gcv <- as.numeric(names(which.min(ridge_grid$GCV)))
cat('GCV lambda:', lambda_gcv)   # expect ~0.5-3 for this simulation

ridge_fit   <- lm.ridge(y ~ ., data = df_sim, lambda = lambda_gcv)
ridge_coefs <- coef(ridge_fit)         # position 1 = intercept
ridge_beta  <- ridge_coefs[-1]


# Residual sigma^2 from the ridge predictions
Xmat       <- as.matrix(df_sim[, -1])
ridge_pred <- as.vector(cbind(1, Xmat) %*% ridge_coefs)
sigma2_est <- mean((df_sim$y - ridge_pred)^2)
sigma_est  <- sqrt(sigma2_est)

# tau from the correspondence: lambda = sigma^2 / tau^2
tau_est <- sigma_est / sqrt(lambda_gcv)

cat('sigma_est:', round(sigma_est, 4))
cat('tau_est:  ', round(tau_est,   4))
cat('Verify — sigma^2/tau^2 =', round(sigma2_est/tau_est^2, 4),
    '== lambda_gcv:', lambda_gcv)



fit_p1 <- brm(
  y ~ .,
  data  = df_sim,
  prior = c(
    # Normal(0, tau) on every slope — this IS the ridge penalty
    set_prior(paste0('normal(0,', round(tau_est, 5), ')'), class = 'b'),
    set_prior('normal(0, 10)',                             class = 'Intercept'),
    # Pin sigma so posterior mean ~ MAP
    set_prior(paste0('normal(', round(sigma_est, 5), ', 0.001)'), class = 'sigma')
  ),
  chains = 2, iter = 2000, warmup = 500,
  seed = 123, backend = 'cmdstanr', refresh = 0
)
