from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"

for d in (OUT_DIR, PLOTS_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

def load_datasets():
    """Load and merge agentic and human datasets locally."""
    agentic = pd.read_parquet(DATA_DIR / "agentic_refactoring_commits.parquet")
    human = pd.read_parquet(DATA_DIR / "baseline_refactoring_commits_normalized.parquet")
    human_pr = pd.read_parquet(DATA_DIR / "baseline_pr_commits.parquet")

    agentic["dataset"] = "Agentic"
    human["dataset"] = "Human"

    df = pd.concat([agentic, human], ignore_index=True)
    df["refactoring_count"] = df["refactoring_count"].fillna(0).astype(int)
    return df

df = load_datasets()
agentic = df[df["dataset"] == "Agentic"]
human = df[df["dataset"] == "Human"]

#Summary per Project
def summarize_per_project(sub_df, label):
    proj = (
        sub_df.groupby(["agent", "full_name"], dropna=False)
        .agg(
            total_commits=("sha", "nunique"),
            refactoring_commits=("has_refactoring", "sum"),
            total_refactorings=("refactoring_count", "sum"),
            mean_refactorings=("refactoring_count", "mean"),
            median_refactorings=("refactoring_count", "median"),
        )
        .reset_index()
    )
    proj["refactoring_rate_%"] = (
        proj["refactoring_commits"] / proj["total_commits"] * 100
    )
    proj["refactors_per_all_commits"] = (
        proj["total_refactorings"] / proj["total_commits"]
    )
    proj["refactors_per_refactoring_commit"] = proj.apply(
        lambda r: r["total_refactorings"] / r["refactoring_commits"]
        if r["refactoring_commits"] > 0 else 0,
        axis=1,
    )
    proj["denominator"] = label
    return proj


agentic_proj = summarize_per_project(agentic, "Observed agentic commits")
human_proj = summarize_per_project(human, "Observed human commits")

proj_summary = pd.concat([agentic_proj, human_proj], ignore_index=True)
proj_summary.to_csv(TABLES_DIR / "per_project_refactoring_rate.csv", index=False)

stats = (
    proj_summary.groupby("agent")[[
        "refactoring_rate_%",
        "refactors_per_all_commits",
        "refactors_per_refactoring_commit"
    ]]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

stats.to_csv(TABLES_DIR / "per_agent_refactoring_stats.csv")

print("\nPer-Agent Refactoring Statistics per Project:")
print(stats.round(3).to_string())


#Refactoring commits/total commits
df_valid = df.copy()
df_valid["refactoring_count"] = pd.to_numeric(df_valid["refactoring_count"], errors="coerce").fillna(0)
df_valid["has_refactoring"] = df_valid["has_refactoring"].astype(bool)

table_commits = (
    df_valid.groupby("agent", dropna=False)
    .agg(
        total_commits=("sha", "nunique"),
        refactoring_commits=("has_refactoring", "sum"),
        total_refactorings=("refactoring_count", "sum"),
    )
    .reset_index()
)

table_commits["refactoring_rate_%"] = (
    table_commits["refactoring_commits"] / table_commits["total_commits"] * 100
)

#Refactors/refactoring commit
table_commits["mean_refactors_per_ref_commit"] = table_commits.apply(
    lambda r: r["total_refactorings"] / r["refactoring_commits"] if r["refactoring_commits"] > 0 else 0,
    axis=1
)

table_commits = table_commits.round(3)

print("\nTable 1 — Commit and Refactoring Rates per Agent:")
print(table_commits.to_string(index=False))

table_commits.to_csv(TABLES_DIR / "per_agent_commit_and_refactoring_rate.csv", index=False)
refactoring_commits = df_valid[df_valid["has_refactoring"]]

table_refactors = (
    refactoring_commits.groupby("agent", dropna=False)["refactoring_count"]
    .agg(["mean", "median", "std", "min", "max", "count"])
    .rename(columns={
        "mean": "mean_refactors_per_ref_commit",
        "median": "median_refactors_per_ref_commit",
        "std": "std_refactors_per_ref_commit",
        "min": "min_refactors_per_ref_commit",
        "max": "max_refactors_per_ref_commit",
        "count": "num_refactoring_commits"
    })
    .reset_index()
)

table_refactors = table_refactors.round(3)

print("\nTable 2 — Refactors per Refactoring Commit (Mean/Median/Std/Min/Max):")
print(table_refactors.to_string(index=False))

table_refactors.to_csv(TABLES_DIR / "per_agent_refactors_per_ref_commit.csv", index=False)

print("\nBoth tables generated and saved successfully.")


#Boxplots
agents = sorted(proj_summary["agent"].dropna().unique())

def make_boxplot(metric_col, title, ylabel, filename, log_scale=False):
    box_data = [
        proj_summary.loc[proj_summary["agent"] == a, metric_col].dropna()
        for a in agents
    ]
    plt.figure(figsize=(10, 5))
    plt.boxplot(box_data, showfliers=False, tick_labels=agents)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Agent")
    if log_scale:
        plt.yscale("log")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()
    print(f"\nSaved plot: {filename}")


#Refactoring rate
make_boxplot(
    "refactoring_rate_%",
    "Per-Project Refactoring Commit Rate by Agent",
    "Refactoring Commit Rate (%)",
    "box_refactoring_rate_per_project.png",
)

#Refactors per all commits
make_boxplot(
    "refactors_per_all_commits",
    "Per-Project Refactors per All Commits by Agent (log scale)",
    "Refactors per Commit (log)",
    "box_refactors_per_all_commits_per_project.png",
    log_scale=True,
)

#Refactors per refactoring commit
make_boxplot(
    "refactors_per_refactoring_commit",
    "Per-Project Refactors per Refactoring Commit by Agent (log scale)",
    "Refactors per Refactoring Commit (log)",
    "box_refactors_per_refactoring_commit_per_project.png",
    log_scale=True,
)
