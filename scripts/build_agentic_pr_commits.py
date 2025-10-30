import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed/agentic_pr_commits.parquet")

print("Loading base datasets...")
repos = pd.read_parquet(RAW / "all_repository.parquet")
prs = pd.read_parquet(RAW / "pull_request.parquet")
commits = pd.read_parquet(RAW / "pr_commits.parquet")

print(f"Repositories: {len(repos):,}, PRs: {len(prs):,}, Commits: {len(commits):,}")

# Filter Java repositories
repos_java = repos[repos["language"].str.lower() == "java"]
print(f"Java repos: {len(repos_java):,}")

# Join PRs with their repositories
prs_merged = prs.merge(repos_java, left_on="repo_id", right_on="id", suffixes=("", "_repo"))
print(f"Java PRs after merge: {len(prs_merged):,}")

# Join commits with PRs
merged = commits.merge(prs_merged, left_on="pr_id", right_on="id", how="inner")

# Keep only AI-agentic PRs
merged = merged[merged["agent"].notna() & (merged["agent"].str.strip() != "")]
print(f"AI-agentic PRs: {len(merged):,}")

# Keep essential columns
final = merged[[
    "sha", "pr_id", "number", "repo_url", "full_name", "language", "agent"
]].drop_duplicates()

OUT.parent.mkdir(parents=True, exist_ok=True)
final.to_parquet(OUT)
print(final.sample(5))
