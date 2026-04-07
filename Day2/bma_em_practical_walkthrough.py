import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

df = pd.read_csv("bma_em_practical_dataset.csv")

model_cols = [
    "model_1_biased_lowvar",
    "model_2_unbiased_noisy",
    "model_3_context_dependent",
    "model_4_bad_spatial",
]

X = df[model_cols].to_numpy()
y = df["y_true"].to_numpy()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def fit_bma_em(
    X,
    y,
    max_iter=500,
    ll_tol=1e-6,
    sigma_floor=1e-6,
    denom_floor=1e-12,
    weight_floor=1e-12,
    verbose=False,
):
    n, m = X.shape
    weights = np.ones(m) / m
    sigmas = np.sqrt(np.mean((X - y[:, None]) ** 2, axis=0))
    sigmas = np.maximum(sigmas, sigma_floor)
    loglik_trace = []

    for it in range(max_iter):
        diff = y[:, None] - X
        gaussian = (
            1.0 / (np.sqrt(2 * np.pi) * sigmas[None, :])
            * np.exp(-0.5 * (diff / sigmas[None, :]) ** 2)
        )
        numerators = weights[None, :] * gaussian
        denominators = np.sum(numerators, axis=1, keepdims=True)
        denominators = np.maximum(denominators, denom_floor)
        responsibilities = numerators / denominators

        Nk = responsibilities.sum(axis=0)
        weights_new = Nk / n
        weights_new = np.maximum(weights_new, weight_floor)
        weights_new = weights_new / weights_new.sum()

        sigmas_new = np.sqrt(
            np.sum(responsibilities * (diff ** 2), axis=0) / np.maximum(Nk, denom_floor)
        )
        sigmas_new = np.maximum(sigmas_new, sigma_floor)

        gaussian_new = (
            1.0 / (np.sqrt(2 * np.pi) * sigmas_new[None, :])
            * np.exp(-0.5 * (diff / sigmas_new[None, :]) ** 2)
        )
        mixture = np.sum(weights_new[None, :] * gaussian_new, axis=1)
        mixture = np.maximum(mixture, denom_floor)
        ll = np.sum(np.log(mixture))
        loglik_trace.append(ll)

        if verbose and (it < 5 or (it + 1) % 20 == 0):
            print(f"iter={it+1:03d} loglik={ll:.6f}")

        if it > 0 and abs(loglik_trace[-1] - loglik_trace[-2]) < ll_tol:
            weights = weights_new
            sigmas = sigmas_new
            break

        weights = weights_new
        sigmas = sigmas_new

    return {
        "weights": weights,
        "sigmas": sigmas,
        "responsibilities": responsibilities,
        "loglik_trace": np.array(loglik_trace),
    }

def predict_bma_mean(X, weights):
    return X @ weights

def predict_bma_variance(X, weights, sigmas):
    mean_pred = predict_bma_mean(X, weights)
    within = np.sum(weights * (sigmas ** 2))
    between = np.sum(weights[None, :] * (X - mean_pred[:, None]) ** 2, axis=1)
    total = within + between
    return mean_pred, within, between, total

print("\n--- Individual model performance ---")
for col in model_cols:
    print(f"{col:28s} RMSE={rmse(y, df[col].to_numpy()):.3f}  R2={r2_score_manual(y, df[col].to_numpy()):.3f}")

y_sma = X.mean(axis=1)
print("\n--- SMA ---")
print(f"RMSE={rmse(y, y_sma):.3f}  R2={r2_score_manual(y, y_sma):.3f}")

fit_all = fit_bma_em(X, y, verbose=True)
weights_all = fit_all["weights"]
sigmas_all = fit_all["sigmas"]
y_bma_all, within_all, between_all, total_all = predict_bma_variance(X, weights_all, sigmas_all)

print("\n--- BMA-EM on all data ---")
for model, w, s in zip(model_cols, weights_all, sigmas_all):
    print(f"{model:28s} weight={w:.3f}  sigma={s:.3f}")
print(f"BMA RMSE={rmse(y, y_bma_all):.3f}  R2={r2_score_manual(y, y_bma_all):.3f}")
print(f"Within variance={within_all:.3f}")
print(f"Mean between variance={between_all.mean():.3f}")

train_mask = df["split"] == "train"
test_mask = df["split"] == "test"

X_train = df.loc[train_mask, model_cols].to_numpy()
y_train = df.loc[train_mask, "y_true"].to_numpy()
X_test = df.loc[test_mask, model_cols].to_numpy()
y_test = df.loc[test_mask, "y_true"].to_numpy()

fit_train = fit_bma_em(X_train, y_train, verbose=False)
weights_train = fit_train["weights"]
sigmas_train = fit_train["sigmas"]

y_sma_train = X_train.mean(axis=1)
y_sma_test = X_test.mean(axis=1)
y_bma_train, within_train, between_train, total_train = predict_bma_variance(X_train, weights_train, sigmas_train)
y_bma_test, within_test, between_test, total_test = predict_bma_variance(X_test, weights_train, sigmas_train)

print("\n--- Spatial generalization ---")
print(f"Train SMA RMSE={rmse(y_train, y_sma_train):.3f} | Train BMA RMSE={rmse(y_train, y_bma_train):.3f}")
print(f"Test  SMA RMSE={rmse(y_test, y_sma_test):.3f} | Test  BMA RMSE={rmse(y_test, y_bma_test):.3f}")

plt.figure(figsize=(6,4))
plt.plot(fit_all["loglik_trace"])
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.title("EM convergence")
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(y, y_sma, alpha=0.7, label="SMA")
plt.scatter(y, y_bma_all, alpha=0.7, label="BMA-EM")
mn = min(y.min(), y_sma.min(), y_bma_all.min())
mx = max(y.max(), y_sma.max(), y_bma_all.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Observed")
plt.ylabel("Prediction")
plt.legend()
plt.title("Full dataset")
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(y_test, y_sma_test, alpha=0.7, label="SMA test")
plt.scatter(y_test, y_bma_test, alpha=0.7, label="BMA-EM test")
mn = min(y_test.min(), y_sma_test.min(), y_bma_test.min())
mx = max(y_test.max(), y_sma_test.max(), y_bma_test.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Observed")
plt.ylabel("Prediction")
plt.legend()
plt.title("Held-out spatial test set")
plt.show()
