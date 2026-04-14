"""
q2.py — Trajectories, Optimization & Selecting K
MSE 555 Assignment 3

Input:
    data/labeled_notes.json        — 40 clients, Patel ground-truth scores
    output/q1/scored_notes.csv     — 80 unlabeled clients, LLM scores

Output:
    output/q2/spaghetti_K{k}.png           — spaghetti plots for K=2..5
    output/q2/plot1_tstar_distributions.png
    output/q2/plot2_savings_vs_Q.png
    output/q2/plot3_optimized_vs_baseline.png
    output/q2/client_clusters.csv
    output/q2/cluster_Q_stars.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ── reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
K_VALUES = [2, 3, 4, 5]
TMAX = 12          # total sessions (1–12)
N_TRANSITIONS = 11 # scores per client (transitions 1→2 .. 11→12)
PLATEAU_FRAC = 0.90

OUT_DIR = Path("output/q2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data loading
# ============================================================================

def load_scores() -> pd.DataFrame:
    """
    Combine labeled (ground-truth) and unlabeled (LLM) scores into one
    DataFrame with columns: client_id, score_1 … score_11.
    """
    # --- labeled: 40 clients, Patel ground-truth ---
    labeled_raw = json.load(open("data/labeled_notes.json"))
    labeled_rows = []
    for rec in labeled_raw:
        scores = rec["scored_progress"]          # list of 11 ints
        row = {"client_id": rec["client_id"]}
        for i, s in enumerate(scores, start=1):
            row[f"score_{i}"] = s
        labeled_rows.append(row)
    df_labeled = pd.DataFrame(labeled_rows)

    # --- unlabeled: 80 clients, LLM scores from Q1 ---
    df_csv = pd.read_csv("output/q1/scored_notes.csv")
    # session column = transition index 1..11, already sorted per client
    unlabeled_rows = []
    for cid, grp in df_csv.groupby("client_id", sort=False):
        grp = grp.sort_values("session")
        row = {"client_id": cid}
        for _, r in grp.iterrows():
            row[f"score_{int(r['session'])}"] = int(r["score"])
        unlabeled_rows.append(row)
    df_unlabeled = pd.DataFrame(unlabeled_rows)

    df = pd.concat([df_labeled, df_unlabeled], ignore_index=True)
    print(f"Total clients: {len(df)}  (labeled={len(df_labeled)}, unlabeled={len(df_unlabeled)})")
    return df


# ============================================================================
# Feature engineering
# ============================================================================

def build_cumulative(df: pd.DataFrame) -> np.ndarray:
    """
    For each client, compute cumulative sum over score_1..score_11.
    Returns array of shape (n_clients, 11).
    """
    score_cols = [f"score_{i}" for i in range(1, 12)]
    raw = df[score_cols].values.astype(float)      # (n, 11)
    cumulative = np.cumsum(raw, axis=1)            # (n, 11)
    return cumulative


def compute_tstar(cumulative: np.ndarray) -> np.ndarray:
    """
    For each client, find t*: the earliest SESSION s in 1..12 where
    cumulative progress >= 90% of total cumulative progress.

    We pad cumulative with a 0 at position 0 so that index s maps to
    session s (sessions 1–12, cumulative[s] = progress through transition s).
      cumulative_padded shape: (n, 12)  indexed [0..11] → sessions 1..12
      cumulative_padded[:, 0]  = 0  (session 1, no transitions yet)
      cumulative_padded[:, s]  = cumsum through transition s

    t* is 1-indexed (session number 1–12).
    """
    n = cumulative.shape[0]
    # pad with 0 at the front → shape (n, 12), index i = session i+1...
    # simpler: padded[i, s] = cumulative progress at session s+1
    padded = np.hstack([np.zeros((n, 1)), cumulative])  # (n, 12), col j = session j+1

    total = padded[:, -1]  # cumulative progress at session 12 = score sum

    tstar = np.full(n, TMAX, dtype=int)  # default: session 12 (never plateau)
    for i in range(n):
        if total[i] == 0:
            tstar[i] = TMAX
            continue
        threshold = PLATEAU_FRAC * total[i]
        # find earliest session s (1..12) where padded[i, s-1] >= threshold
        for s in range(1, TMAX + 1):
            if padded[i, s - 1] >= threshold:
                tstar[i] = s
                break

    return tstar


# ============================================================================
# Newsvendor policy
# ============================================================================

def compute_policy(tstar: np.ndarray) -> dict:
    """
    For a set of clients (one cluster), compute:
      F(Q) = P(t* <= Q)  for Q = 1..12
      E[savings](Q) = F(Q) * (12 - Q)
      Q* = argmax E[savings](Q)

    Returns dict with keys: Q_values, F, E_savings, Q_star, E_saved_at_Qstar.
    """
    Q_values = np.arange(1, TMAX + 1)
    n = len(tstar)
    F = np.array([(tstar <= Q).sum() / n for Q in Q_values])
    E_savings = F * (TMAX - Q_values)

    # argmax: if tie, prefer smaller Q (earlier reassessment)
    Q_star_idx = int(np.argmax(E_savings))
    Q_star = Q_values[Q_star_idx]
    E_saved = E_savings[Q_star_idx]

    return {
        "Q_values": Q_values,
        "F": F,
        "E_savings": E_savings,
        "Q_star": int(Q_star),
        "E_saved_at_Qstar": float(E_saved),
    }


# ============================================================================
# K-Means clustering
# ============================================================================

def run_kmeans(cumulative: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(cumulative)
    return labels


# ============================================================================
# Plots
# ============================================================================

CLUSTER_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed"]


def plot_spaghetti(cumulative: np.ndarray, labels: np.ndarray, k: int) -> None:
    """One subplot per cluster showing individual trajectories + cluster mean."""
    x = np.arange(1, N_TRANSITIONS + 1)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]

    for c in range(k):
        ax = axes[c]
        mask = labels == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

        for traj in cumulative[mask]:
            ax.plot(x, traj, color=color, alpha=0.18, linewidth=0.8)

        mean_traj = cumulative[mask].mean(axis=0)
        ax.plot(x, mean_traj, color=color, linewidth=2.5, label="Mean")

        ax.set_title(f"Cluster {c}  (n={mask.sum()})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Transition")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Cumulative Progress")
    fig.suptitle(f"Cumulative Trajectories — K={k}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / f"spaghetti_K{k}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_tstar_distributions(tstar: np.ndarray, labels: np.ndarray, k: int) -> None:
    """Histogram of t* per cluster, shared x-axis."""
    fig, axes = plt.subplots(k, 1, figsize=(8, 2.5 * k), sharex=True)
    if k == 1:
        axes = [axes]

    for c in range(k):
        ax = axes[c]
        mask = labels == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.hist(tstar[mask], bins=np.arange(0.5, TMAX + 1.5), color=color,
                edgecolor="white", alpha=0.85)
        ax.set_ylabel(f"Cluster {c}\n(n={mask.sum()})", fontsize=10)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3)

    axes[-1].set_xlabel("t* (stopping session)", fontsize=11)
    axes[-1].set_xticks(range(1, TMAX + 1))
    fig.suptitle("Distribution of t* by Cluster", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "plot1_tstar_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_savings_vs_Q(policies: list[dict], labels: np.ndarray, k: int) -> None:
    """E[savings](Q) for each cluster, with Q* markers."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for c, pol in enumerate(policies):
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        mask = labels == c
        ax.plot(pol["Q_values"], pol["E_savings"], color=color, linewidth=2,
                label=f"Cluster {c} (n={mask.sum()}, Q*={pol['Q_star']})")
        ax.axvline(pol["Q_star"], color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.scatter([pol["Q_star"]], [pol["E_saved_at_Qstar"]],
                   color=color, s=80, zorder=5)

    ax.set_xlabel("Reassessment Session Q", fontsize=12)
    ax.set_ylabel("E[Sessions Saved per Child]", fontsize=12)
    ax.set_title("Expected Savings vs Reassessment Session", fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, TMAX + 1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "plot2_savings_vs_Q.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_optimized_vs_baseline(
    tstar: np.ndarray, labels: np.ndarray, policies: list[dict], k: int
) -> None:
    """Grouped bar: E[savings/child] at Q* vs naive baseline (round(mean t*))."""
    clusters = list(range(k))
    opt_savings = []
    base_savings = []

    for c, pol in enumerate(policies):
        mask = labels == c
        ts = tstar[mask]
        n = mask.sum()

        # Optimal
        opt_savings.append(pol["E_saved_at_Qstar"])

        # Baseline: Q_base = round(mean t*), clipped to 1..11 (savings = 0 at Q=12)
        Q_base = int(np.round(ts.mean()))
        Q_base = max(1, min(TMAX - 1, Q_base))
        F_base = (ts <= Q_base).sum() / n
        base_savings.append(F_base * (TMAX - Q_base))

    x = np.arange(k)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_opt = ax.bar(x - width / 2, opt_savings, width, label="Optimal Q*",
                      color=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in clusters],
                      alpha=0.9)
    bars_base = ax.bar(x + width / 2, base_savings, width, label="Naive baseline",
                       color=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in clusters],
                       alpha=0.45, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_ylabel("E[Sessions Saved per Child]", fontsize=11)
    ax.set_title("Optimized Q* vs Naive Baseline", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # annotate total gain
    total_opt = sum(opt_savings)
    total_base = sum(base_savings)
    ax.annotate(
        f"Total gain over baseline: {total_opt - total_base:+.2f} sessions",
        xy=(0.5, 0.95), xycoords="axes fraction", ha="center", fontsize=10,
        color="#15803d"
    )

    plt.tight_layout()
    path = OUT_DIR / "plot3_optimized_vs_baseline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# K-selection comparison table
# ============================================================================

def print_k_comparison(
    cumulative: np.ndarray,
    tstar: np.ndarray,
    df: pd.DataFrame,
) -> tuple[int, np.ndarray, list[dict]]:
    """
    For K = 2..5, compute policy outcomes and print a comparison table.
    Returns (best_K, best_labels).

    K selection criterion: prefer the K where clusters produce the most
    distinct Q* values (clustering is only useful if policies differ), with
    a tiebreak on total expected savings.
    """
    print("\n" + "=" * 72)
    print("K-SELECTION: Policy Outcome Comparison")
    print("=" * 72)

    results = {}
    for k in K_VALUES:
        labels = run_kmeans(cumulative, k)
        policies = []
        for c in range(k):
            mask = labels == c
            pol = compute_policy(tstar[mask])
            pol["size"] = int(mask.sum())
            pol["cluster"] = c
            policies.append(pol)

        q_stars = [p["Q_star"] for p in policies]
        n_distinct = len(set(q_stars))
        total_savings = sum(p["E_saved_at_Qstar"] * p["size"] for p in policies)
        avg_savings = total_savings / len(tstar)

        results[k] = {
            "labels": labels,
            "policies": policies,
            "n_distinct_Qstar": n_distinct,
            "total_savings": total_savings,
            "avg_savings": avg_savings,
        }

        print(f"\n  K={k}  ({n_distinct}/{k} distinct Q* values, "
              f"avg savings={avg_savings:.3f})")
        print(f"  {'Cluster':>8} {'Size':>6} {'Q*':>4} {'E[saved/child]':>15} {'% saved':>8}")
        print(f"  {'-'*8} {'-'*6} {'-'*4} {'-'*15} {'-'*8}")
        for p in policies:
            pct = p["E_saved_at_Qstar"] / TMAX * 100
            print(f"  {p['cluster']:>8} {p['size']:>6} {p['Q_star']:>4} "
                  f"{p['E_saved_at_Qstar']:>15.3f} {pct:>7.1f}%")

    # --- pick best K ---
    # Select K with maximum distinct Q* values, tiebreak by smallest K, then avg savings
    best_k = max(results.keys(), key=lambda k: (results[k]["n_distinct_Qstar"], -k, results[k]["avg_savings"]))
    print(f"\n>>> Selected K={best_k} "
          f"({results[best_k]['n_distinct_Qstar']} distinct Q* values, "
          f"avg savings={results[best_k]['avg_savings']:.3f}) "
          f"[criterion: max distinct Q*, tiebreak smallest K, then avg savings]")

    return best_k, results[best_k]["labels"], results[best_k]["policies"]


# ============================================================================
# Summary table
# ============================================================================

def print_summary_table(policies: list[dict], k: int) -> None:
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    header = f"{'Cluster':>8} {'Size':>6} {'Q*':>4} {'E[saved/child]':>15} {'% sessions saved':>17}"
    print(header)
    print("-" * 65)

    total_weighted = 0.0
    total_n = 0
    for p in policies:
        pct = p["E_saved_at_Qstar"] / TMAX * 100
        print(f"{p['cluster']:>8} {p['size']:>6} {p['Q_star']:>4} "
              f"{p['E_saved_at_Qstar']:>15.3f} {pct:>16.1f}%")
        total_weighted += p["E_saved_at_Qstar"] * p["size"]
        total_n += p["size"]

    overall = total_weighted / total_n
    overall_pct = overall / TMAX * 100
    print("-" * 65)
    print(f"{'Total':>8} {total_n:>6} {'--':>4} {overall:>15.3f} {overall_pct:>16.1f}%")
    print("=" * 65)


# ============================================================================
# Save outputs for Q3
# ============================================================================

def save_outputs(
    df: pd.DataFrame, labels: np.ndarray, policies: list[dict]
) -> None:
    # client_clusters.csv
    clusters_df = pd.DataFrame({
        "client_id": df["client_id"].values,
        "cluster": labels,
    })
    path1 = OUT_DIR / "client_clusters.csv"
    clusters_df.to_csv(path1, index=False)
    print(f"Saved: {path1}")

    # cluster_Q_stars.csv
    qstar_df = pd.DataFrame([
        {
            "cluster": p["cluster"],
            "Q_star": p["Q_star"],
            "E_saved_per_child": round(p["E_saved_at_Qstar"], 4),
        }
        for p in policies
    ])
    path2 = OUT_DIR / "cluster_Q_stars.csv"
    qstar_df.to_csv(path2, index=False)
    print(f"Saved: {path2}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # --- load & build features ---
    df = load_scores()
    cumulative = build_cumulative(df)
    tstar = compute_tstar(cumulative)

    print(f"\nt* summary: min={tstar.min()}, max={tstar.max()}, "
          f"mean={tstar.mean():.1f}, median={np.median(tstar):.1f}")
    print(f"Distribution: { {s: int((tstar == s).sum()) for s in range(1, TMAX+1)} }")

    # --- spaghetti plots for all K ---
    print("\nGenerating spaghetti plots...")
    for k in K_VALUES:
        labels_k = run_kmeans(cumulative, k)
        plot_spaghetti(cumulative, labels_k, k)

    # --- K selection ---
    best_k, best_labels, best_policies = print_k_comparison(cumulative, tstar, df)

    # --- final K plots ---
    print(f"\nGenerating final plots for K={best_k}...")
    plot_tstar_distributions(tstar, best_labels, best_k)
    plot_savings_vs_Q(best_policies, best_labels, best_k)
    plot_optimized_vs_baseline(tstar, best_labels, best_policies, best_k)

    # --- summary ---
    print_summary_table(best_policies, best_k)

    # --- save CSVs for Q3 ---
    save_outputs(df, best_labels, best_policies)

    # --- Q2(f) placeholder ---
    print("\nQ2(f) — Written discussion to be completed in report.")


if __name__ == "__main__":
    main()
