import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ktm import processing_data


@dataclass
class EMResult:
    n_components: int
    pi: np.ndarray
    beta_mu: np.ndarray
    gamma_sigma: np.ndarray
    train_nll: float
    train_rmse: float
    train_mae: float
    converged: bool
    n_iter: int


def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    sw = np.sqrt(np.maximum(w, 1e-12))
    xw = x * sw[:, None]
    yw = y * sw
    xtx = xw.T @ xw
    if ridge > 0:
        xtx = xtx + ridge * np.eye(xtx.shape[0])
    xty = xw.T @ yw
    return np.linalg.solve(xtx, xty)


def _component_log_sigma(
    x_sigma: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    return x_sigma @ gamma


def _component_sigma(
    x_sigma: np.ndarray,
    gamma: np.ndarray,
    sigma_floor: float,
) -> np.ndarray:
    log_s = _component_log_sigma(x_sigma, gamma)
    s = np.exp(log_s)
    return np.maximum(s, sigma_floor)


def _optimize_gamma(
    x_sigma: np.ndarray,
    resid: np.ndarray,
    resp: np.ndarray,
    gamma0: np.ndarray,
    *,
    sigma_floor: float,
    gamma_ridge: float,
) -> np.ndarray:
    # Minimize weighted gaussian nll wrt gamma where sigma = exp(X gamma)
    # L = sum_i r_i [log sigma_i + 0.5 * (resid_i^2 / sigma_i^2)]
    def obj(g: np.ndarray) -> float:
        sigma = _component_sigma(x_sigma, g, sigma_floor=sigma_floor)
        z2 = (resid / sigma) ** 2
        v = np.sum(resp * (np.log(sigma) + 0.5 * z2))
        if gamma_ridge > 0:
            v += 0.5 * gamma_ridge * float(np.sum(g ** 2))
        return float(v)

    out = minimize(obj, gamma0, method="L-BFGS-B")
    if out.success:
        return out.x
    return gamma0


def _init_params(
    x_mu: np.ndarray,
    y: np.ndarray,
    k: int,
    q_sigma: int,
    seed: int,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    p = x_mu.shape[1]

    pi = np.full(k, 1.0 / k, dtype=float)
    beta_mu = np.zeros((k, p), dtype=float)
    gamma_sigma = np.zeros((k, q_sigma), dtype=float)

    if k == 1:
        beta_mu[0] = np.linalg.lstsq(x_mu, y, rcond=None)[0]
        resid = y - x_mu @ beta_mu[0]
        s0 = max(float(np.std(resid)), sigma_floor)
        gamma_sigma[0, 0] = float(np.log(s0))
        return pi, beta_mu, gamma_sigma

    # quantile partition for robust init
    qs = np.quantile(y, np.linspace(0, 1, k + 1))
    comp = np.digitize(y, qs[1:-1], right=True)
    if len(np.unique(comp)) < k:
        comp = rng.integers(0, k, size=n)

    for j in range(k):
        mask = comp == j
        nj = int(mask.sum())
        if nj < p:
            beta_mu[j] = np.linalg.lstsq(x_mu, y, rcond=None)[0]
            resid = y - x_mu @ beta_mu[j]
            s0 = max(float(np.std(resid)), sigma_floor)
            gamma_sigma[j, 0] = float(np.log(s0))
            continue
        beta_mu[j] = np.linalg.lstsq(x_mu[mask], y[mask], rcond=None)[0]
        resid = y[mask] - x_mu[mask] @ beta_mu[j]
        s0 = max(float(np.std(resid)), sigma_floor)
        gamma_sigma[j, 0] = float(np.log(s0))
        pi[j] = nj / n

    pi = np.maximum(pi, 1e-8)
    pi = pi / pi.sum()
    return pi, beta_mu, gamma_sigma


def evaluate_gmm_regression(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    pi: np.ndarray,
    beta_mu: np.ndarray,
    gamma_sigma: np.ndarray,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
) -> dict[str, float]:
    x_mu = _poly_design(theta, mu_degree)
    x_sigma = _poly_design(theta, sigma_degree)
    y = delta.astype(float)
    mu = x_mu @ beta_mu.T
    k = mu.shape[1]
    sigma = np.column_stack(
        [
            _component_sigma(x_sigma, gamma_sigma[j], sigma_floor=sigma_floor)
            for j in range(k)
        ]
    )
    sig2 = np.maximum(sigma ** 2, sigma_floor ** 2)
    logp = (
        np.log(np.maximum(pi, 1e-12))[None, :]
        - 0.5 * np.log(2.0 * np.pi * sig2)
        - 0.5 * ((y[:, None] - mu) ** 2) / sig2
    )
    log_marg = logsumexp(logp, axis=1)
    nll = -float(np.mean(log_marg))
    y_hat = mu @ pi
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    mae = float(np.mean(np.abs(y - y_hat)))
    sigma_floor_frac = float(np.mean(sigma <= (sigma_floor + 1e-12)))
    return {
        "nll": nll,
        "rmse": rmse,
        "mae": mae,
        "sigma_floor_frac": sigma_floor_frac,
    }


def fit_gaussian_mixture_regression_em(
    theta: np.ndarray,
    delta: np.ndarray,
    n_components: int,
    *,
    mu_degree: int = 1,
    sigma_degree: int = 1,
    max_iter: int = 200,
    tol: float = 1e-5,
    sigma_floor: float = 0.1,
    beta_ridge: float = 1e-8,
    gamma_ridge: float = 1e-4,
    seed: int = 42,
) -> EMResult:
    """
    EM for:
      p(delta|theta)=sum_k pi_k N(delta|mu_k(theta), sigma_k(theta)^2)

    with:
      mu_k(theta) polynomial degree mu_degree
      log sigma_k(theta) polynomial degree sigma_degree
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if mu_degree < 1:
        raise ValueError("mu_degree must be >= 1")
    if sigma_degree < 0:
        raise ValueError("sigma_degree must be >= 0")

    x_mu = _poly_design(theta, mu_degree)
    x_sigma = _poly_design(theta, sigma_degree)
    y = delta.astype(float)
    n = len(y)
    k = int(n_components)

    pi, beta_mu, gamma_sigma = _init_params(
        x_mu,
        y,
        k=k,
        q_sigma=x_sigma.shape[1],
        seed=seed,
        sigma_floor=sigma_floor,
    )

    prev_nll = np.inf
    converged = False

    for it in range(1, max_iter + 1):
        # E-step
        mu = x_mu @ beta_mu.T
        sigma = np.column_stack(
            [
                _component_sigma(x_sigma, gamma_sigma[j], sigma_floor=sigma_floor)
                for j in range(k)
            ]
        )
        sig2 = np.maximum(sigma ** 2, sigma_floor ** 2)
        logp = (
            np.log(np.maximum(pi, 1e-12))[None, :]
            - 0.5 * np.log(2.0 * np.pi * sig2)
            - 0.5 * ((y[:, None] - mu) ** 2) / sig2
        )
        log_denom = logsumexp(logp, axis=1, keepdims=True)
        resp = np.exp(logp - log_denom)
        nll = -float(np.mean(log_denom))

        # M-step
        nk = resp.sum(axis=0)
        pi = np.maximum(nk / n, 1e-12)
        pi = pi / pi.sum()

        for j in range(k):
            # beta update with heteroscedastic weights: r / sigma^2
            w_beta = resp[:, j] / np.maximum(sig2[:, j], sigma_floor ** 2)
            beta_mu[j] = _weighted_lstsq(x_mu, y, w_beta, ridge=beta_ridge)

            # gamma update via numeric optimization
            resid = y - x_mu @ beta_mu[j]
            gamma_sigma[j] = _optimize_gamma(
                x_sigma,
                resid,
                resp[:, j],
                gamma_sigma[j],
                sigma_floor=sigma_floor,
                gamma_ridge=gamma_ridge,
            )

        if abs(prev_nll - nll) < tol:
            converged = True
            prev_nll = nll
            break
        prev_nll = nll

    train_eval = evaluate_gmm_regression(
        theta,
        delta,
        pi=pi,
        beta_mu=beta_mu,
        gamma_sigma=gamma_sigma,
        mu_degree=mu_degree,
        sigma_degree=sigma_degree,
        sigma_floor=sigma_floor,
    )
    return EMResult(
        n_components=k,
        pi=pi,
        beta_mu=beta_mu,
        gamma_sigma=gamma_sigma,
        train_nll=float(train_eval["nll"]),
        train_rmse=float(train_eval["rmse"]),
        train_mae=float(train_eval["mae"]),
        converged=converged,
        n_iter=it,
    )


def fit_best_of_restarts(
    theta_train: np.ndarray,
    delta_train: np.ndarray,
    n_components: int,
    *,
    mu_degree: int,
    sigma_degree: int,
    max_iter: int,
    tol: float,
    sigma_floor: float,
    beta_ridge: float,
    gamma_ridge: float,
    restarts: int,
    seed: int,
) -> tuple[EMResult, int]:
    best = None
    best_r = -1
    for r in range(restarts):
        out = fit_gaussian_mixture_regression_em(
            theta_train,
            delta_train,
            n_components=n_components,
            mu_degree=mu_degree,
            sigma_degree=sigma_degree,
            max_iter=max_iter,
            tol=tol,
            sigma_floor=sigma_floor,
            beta_ridge=beta_ridge,
            gamma_ridge=gamma_ridge,
            seed=seed + 997 * r + 13 * n_components,
        )
        if best is None or out.train_nll < best.train_nll:
            best = out
            best_r = r
    return best, best_r


def _build_subsets_by_order(df: pd.DataFrame, min_obs_per_subset: int) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for order_val, g in df.groupby("order_sequence", sort=True):
        if len(g) < min_obs_per_subset:
            continue
        out.append((f"order_{int(order_val)}", g.copy()))
    return out


def _build_subsets_by_bins(df: pd.DataFrame, n_bins: int, min_obs_per_subset: int) -> list[tuple[str, pd.DataFrame]]:
    work = df.copy()
    work["order_bin"] = pd.qcut(work["order_sequence"], q=n_bins, labels=False, duplicates="drop")
    out = []
    for b, g in work.groupby("order_bin", sort=True):
        if len(g) < min_obs_per_subset:
            continue
        lo = int(g["order_sequence"].min())
        hi = int(g["order_sequence"].max())
        out.append((f"bin_{int(b)}_order_{lo}_{hi}", g.copy()))
    return out


def _parse_components(text: str) -> list[int]:
    vals = sorted(set(int(s.strip()) for s in str(text).split(",") if s.strip()))
    if not vals:
        raise ValueError("No components provided")
    if min(vals) < 1:
        raise ValueError("All components must be >= 1")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="KTM Gaussian mixture regression EM with sigma(theta) support."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_processed.csv")
    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="challenge_id")
    parser.add_argument("--correct-col", type=str, default="outcome")
    parser.add_argument("--skill-col", type=str, default="skill_id")
    parser.add_argument("--answer-order-col", type=str, default="answer_number")
    parser.add_argument("--sample-users", type=int, default=10000)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-mode", type=str, default="order", choices=["order", "bin"])
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--min-obs-per-subset", type=int, default=500)
    parser.add_argument("--components", type=str, default="1,2,3,4")
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--sigma-floor", type=float, default=0.1)
    parser.add_argument("--beta-ridge", type=float, default=1e-6)
    parser.add_argument("--gamma-ridge", type=float, default=1e-4)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--out-csv", type=str, default="pix_mapping/ktm_gmm_em_subsets.csv")
    parser.add_argument("--out-summary-csv", type=str, default="pix_mapping/ktm_gmm_em_summary.csv")
    args = parser.parse_args()

    components = _parse_components(args.components)
    if args.mu_degree < 1:
        raise ValueError("--mu-degree must be >= 1")
    if args.sigma_degree < 0:
        raise ValueError("--sigma-degree must be >= 0")
    if not (0.0 < args.val_frac < 0.5):
        raise ValueError("--val-frac must be in (0, 0.5)")

    usecols = [args.user_col, args.item_col, args.correct_col, args.answer_order_col]
    if args.skill_col:
        usecols.append(args.skill_col)
    df = pd.read_csv(args.data, usecols=usecols).dropna(
        subset=[args.user_col, args.item_col, args.correct_col]
    )

    if args.sample_users > 0 and args.sample_users < df[args.user_col].nunique():
        rng = np.random.default_rng(args.seed)
        users = df[args.user_col].dropna().unique()
        keep = set(rng.choice(users, size=args.sample_users, replace=False).tolist())
        df = df[df[args.user_col].isin(keep)].copy()
    if args.sample_rows > 0 and args.sample_rows < len(df):
        df = df.sample(n=args.sample_rows, random_state=args.seed).copy()

    df = df.rename(
        columns={
            args.user_col: "user",
            args.item_col: "item",
            args.correct_col: "correct",
            args.answer_order_col: "answer_number",
        }
    )
    if args.skill_col and args.skill_col in df.columns:
        df = df.rename(columns={args.skill_col: "skill"})

    df_proc = processing_data(
        df,
        user_col="user",
        item_col="item",
        correct_col="correct",
        skill_col="skill",
        order_cols=["answer_number"],
        reduce=False,
        rank_start=0,
        fit_intercept=False,
        center_latents=True,
        target_std=1.0,
        clip_quantiles=(0.01, 0.99),
    )

    if args.subset_mode == "order":
        subsets = _build_subsets_by_order(df_proc, args.min_obs_per_subset)
    else:
        subsets = _build_subsets_by_bins(df_proc, args.n_bins, args.min_obs_per_subset)

    rng = np.random.default_rng(args.seed)
    rows = []
    for subset_name, g in subsets:
        theta = g["proficiency"].to_numpy(dtype=float)
        delta = g["difficulties"].to_numpy(dtype=float)
        n_obs = len(g)

        idx = np.arange(n_obs, dtype=np.int64)
        rng.shuffle(idx)
        n_val = max(1, int(n_obs * args.val_frac))
        n_train = n_obs - n_val
        if n_train < max(args.mu_degree + args.sigma_degree + 4, 12):
            continue
        tr = idx[:n_train]
        va = idx[n_train:]
        theta_tr, delta_tr = theta[tr], delta[tr]
        theta_va, delta_va = theta[va], delta[va]

        for k in components:
            res, best_restart = fit_best_of_restarts(
                theta_tr,
                delta_tr,
                n_components=k,
                mu_degree=args.mu_degree,
                sigma_degree=args.sigma_degree,
                max_iter=args.max_iter,
                tol=args.tol,
                sigma_floor=args.sigma_floor,
                beta_ridge=args.beta_ridge,
                gamma_ridge=args.gamma_ridge,
                restarts=args.restarts,
                seed=args.seed + 31 * k,
            )
            ev_tr = evaluate_gmm_regression(
                theta_tr,
                delta_tr,
                pi=res.pi,
                beta_mu=res.beta_mu,
                gamma_sigma=res.gamma_sigma,
                mu_degree=args.mu_degree,
                sigma_degree=args.sigma_degree,
                sigma_floor=args.sigma_floor,
            )
            ev_va = evaluate_gmm_regression(
                theta_va,
                delta_va,
                pi=res.pi,
                beta_mu=res.beta_mu,
                gamma_sigma=res.gamma_sigma,
                mu_degree=args.mu_degree,
                sigma_degree=args.sigma_degree,
                sigma_floor=args.sigma_floor,
            )

            p_mu = args.mu_degree + 1
            p_sig = args.sigma_degree + 1
            k_params = k * (p_mu + p_sig) + (k - 1)
            total_nll_train = ev_tr["nll"] * n_train
            aic_like = 2 * k_params + 2 * total_nll_train
            bic_like = np.log(max(n_train, 2)) * k_params + 2 * total_nll_train

            rec = {
                "subset": subset_name,
                "n_obs": int(n_obs),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_components": int(k),
                "mu_degree": int(args.mu_degree),
                "sigma_degree": int(args.sigma_degree),
                "train_nll": float(ev_tr["nll"]),
                "val_nll": float(ev_va["nll"]),
                "train_rmse": float(ev_tr["rmse"]),
                "val_rmse": float(ev_va["rmse"]),
                "train_mae": float(ev_tr["mae"]),
                "val_mae": float(ev_va["mae"]),
                "sigma_floor_frac_train": float(ev_tr["sigma_floor_frac"]),
                "sigma_floor_frac_val": float(ev_va["sigma_floor_frac"]),
                "converged": bool(res.converged),
                "n_iter": int(res.n_iter),
                "best_restart": int(best_restart),
                "sigma_floor": float(args.sigma_floor),
                "aic_like": float(aic_like),
                "bic_like": float(bic_like),
            }
            for j in range(k):
                rec[f"pi_{j}"] = float(res.pi[j])
                for m in range(p_mu):
                    rec[f"beta_mu_{j}_{m}"] = float(res.beta_mu[j, m])
                for m in range(p_sig):
                    rec[f"gamma_sigma_{j}_{m}"] = float(res.gamma_sigma[j, m])
            rows.append(rec)

    out_df = pd.DataFrame(rows).sort_values(["n_components", "subset"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    summary_rows = []
    for k, z in out_df.groupby("n_components"):
        summary_rows.append(
            {
                "n_components": int(k),
                "subsets": int(z["subset"].nunique()),
                "obs_total": int(z["n_obs"].sum()),
                "train_obs_total": int(z["n_train"].sum()),
                "val_obs_total": int(z["n_val"].sum()),
                "train_nll_weighted": float((z["train_nll"] * z["n_train"]).sum() / z["n_train"].sum()),
                "val_nll_weighted": float((z["val_nll"] * z["n_val"]).sum() / z["n_val"].sum()),
                "train_rmse_weighted": float((z["train_rmse"] * z["n_train"]).sum() / z["n_train"].sum()),
                "val_rmse_weighted": float((z["val_rmse"] * z["n_val"]).sum() / z["n_val"].sum()),
                "aic_like_total": float(z["aic_like"].sum()),
                "bic_like_total": float(z["bic_like"].sum()),
                "sigma_floor_frac_train_mean": float(z["sigma_floor_frac_train"].mean()),
                "sigma_floor_frac_val_mean": float(z["sigma_floor_frac_val"].mean()),
                "all_converged": bool(z["converged"].all()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("n_components").reset_index(drop=True)
    summary.to_csv(args.out_summary_csv, index=False)

    print(
        f"rows_input={len(df)} rows_processed={len(df_proc)} subsets={len(subsets)} "
        f"components={components} mode={args.subset_mode} mu_degree={args.mu_degree} "
        f"sigma_degree={args.sigma_degree}"
    )
    print(f"saved_csv={os.path.abspath(args.out_csv)}")
    print(f"saved_summary={os.path.abspath(args.out_summary_csv)}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
