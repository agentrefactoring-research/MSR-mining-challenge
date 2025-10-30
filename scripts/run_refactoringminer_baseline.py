import subprocess
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_pr_commits.parquet"
REFMINER_BIN = PROJECT_ROOT / "tools" / "RefactoringMiner-3.0.11"
REPOS_DIR = PROJECT_ROOT / "repos_baseline"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "refminer_baseline_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUTPUT = RESULTS_DIR / "refminer_all_baseline.json"

REFMINER_CMD_BASE = [
    "java", "-cp",
    f"{REFMINER_BIN}/bin;{REFMINER_BIN}/lib/*",
    "org.refactoringminer.RefactoringMiner",
    "-c"
]

print(f"Loading baseline commits from {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
num_prs = df["pr_id"].nunique()
num_repos = df["full_name"].nunique()
print(f"Loaded {len(df)} commits from {num_prs} PRs across {num_repos} repos.")

#Counters
successful_commits = 0
failed_commits = []
successful_repos = set()
all_results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing commits"):
    repo_name = row["full_name"].split("/")[-1]
    repo_path = REPOS_DIR / repo_name
    sha = row["sha"]

    if not repo_path.exists():
        print(f"Missing repo: {repo_name}, skipping {sha[:8]}")
        continue

    temp_json = RESULTS_DIR / "temp_commit.json"
    cmd = REFMINER_CMD_BASE + [str(repo_path), sha, "-json", str(temp_json)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if temp_json.exists():
            with open(temp_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_results.extend(data.get("commits", []))
            temp_json.unlink()

        successful_commits += 1
        successful_repos.add(repo_name)
        print(f"✅ Analyzed {repo_name} ({sha[:8]})")

    except subprocess.CalledProcessError:
        failed_commits.append((repo_name, sha))
        print(f"❌ Failed for {repo_name} ({sha[:8]})")
        continue

with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump({"commits": all_results}, f, indent=2)

#Summary
print(f"Total successful commits: {successful_commits}")
print(f"Total repositories analyzed: {len(successful_repos)}")
print(f"Total failed commits: {len(failed_commits)}")
print(f"Results saved to {FINAL_OUTPUT}")
