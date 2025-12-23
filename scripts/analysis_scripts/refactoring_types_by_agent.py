from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" 
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_PLOTS = PROJECT_ROOT / "outputs" / "plots"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    
# File names
AGENTIC_COMMITS = DATA / "agentic_refactoring_commits.parquet"
AGENTIC_REFACTS = DATA / "agentic_refactorings.parquet"
HUMAN_COMMITS   = DATA / "baseline_refactoring_commits.parquet"
HUMAN_REFACTS   = DATA / "baseline_refactorings.parquet"

plt.rcParams.update({"figure.dpi": 140})
sns.set_theme(style="whitegrid", context="talk")

agentic = pd.read_parquet(AGENTIC_COMMITS)
agentic_ref = pd.read_parquet(AGENTIC_REFACTS)
human = pd.read_parquet(HUMAN_COMMITS)
human_ref = pd.read_parquet(HUMAN_REFACTS)


def _norm_sha(df):
    # Check for variants and rename to 'sha'
    rename_map = {col: 'sha' for col in df.columns if col.lower() in ['sha', 'commit_sha']}
    df = df.rename(columns=rename_map)

    if 'sha' in df.columns:
        df['sha'] = df['sha'].astype(str).str.lower().str.strip()
    else:
        print(f"'sha' column not found in dataframe. Columns: {df.columns.tolist()}")
    return df

agentic = _norm_sha(agentic)
agentic_ref = _norm_sha(agentic_ref)
human = _norm_sha(human)
human_ref = _norm_sha(human_ref)



human_ref = human_ref.merge(
    human[["sha", "full_name"]].drop_duplicates(), 
    on="sha", 
    how="left"
)
human_ref["agent"] = "Human"


if "agent" not in agentic_ref.columns:
    agentic_ref = agentic_ref.merge(
        agentic[["sha", "agent", "full_name"]].drop_duplicates(), 
        on="sha", 
        how="left"
    )


ref_events = pd.concat(
    [
        agentic_ref[["sha", "full_name", "agent", "refactoring_type"]].assign(dataset="Agentic"),
        human_ref[["sha", "full_name", "agent", "refactoring_type"]].assign(dataset="Human"),
    ],
    ignore_index=True
).dropna(subset=["refactoring_type"])


ref_types_by_agent = (
    ref_events.groupby(["agent", "refactoring_type"]).size()
              .reset_index(name="count")
)

ref_types_by_agent["agent_total"] = ref_types_by_agent.groupby("agent")["count"].transform("sum")
ref_types_by_agent["share_pct"] = ref_types_by_agent["count"] / ref_types_by_agent["agent_total"] * 100

ref_types_by_agent.sort_values(["agent", "share_pct"], ascending=[True, False]).to_csv(
    OUT_TABLES / "INFLATED_refactor_types_by_agent.csv", index=False
)


ref_types_by_agent = (
    ref_events.groupby(["agent", "refactoring_type"]).size()
              .reset_index(name="count")
)

ref_types_by_agent["agent_total"] = ref_types_by_agent.groupby("agent")["count"].transform("sum")
ref_types_by_agent["share_pct"] = ref_types_by_agent["count"] / ref_types_by_agent["agent_total"] * 100



intensity_per_commit = (
    ref_events.groupby(["agent", "sha"]).size()
              .reset_index(name="ref_count_per_commit")
)


agent_stats = (
    intensity_per_commit.groupby("agent")["ref_count_per_commit"]
    .agg(["mean", "std", "median", "min", "max"])
    .reset_index()
)


agent_stats.columns = ["agent", "mean_ref", "std_ref", "median_ref", "min_ref", "max_ref"]


agent_stats.to_csv(OUT_TABLES / "agent_intensity_statistics.csv", index=False)


print(agent_stats.to_string(index=False))


ref_types_by_agent.sort_values(["agent", "share_pct"], ascending=[True, False]).to_csv(
    OUT_TABLES / "INFLATED_refactor_types_by_agent.csv", index=False
)

print("\n File finished execution with Standard Deviation.")
