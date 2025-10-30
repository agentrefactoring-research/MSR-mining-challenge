import os
import time
import subprocess
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CLONE_DIR = PROJECT_ROOT / "repos_forks"

for d in [DATA_PROCESSED, CLONE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PULL_REQUESTS = DATA_RAW / "pull_request.parquet"
JAVA_COMMITS = DATA_PROCESSED / "agentic_pr_commits.parquet"

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise EnvironmentError("Please set GITHUB_TOKEN in your environment.")

HEADERS = {"Authorization": f"token {TOKEN}"}

print("Loading PR and commit data...")
pulls = pd.read_parquet(PULL_REQUESTS)
java_commits = pd.read_parquet(JAVA_COMMITS)

java_pr_ids = set(java_commits["pr_id"].unique())
pulls = pulls[pulls["id"].isin(java_pr_ids)]
print(f"Found {len(pulls)} matching Java PRs across {pulls['repo_url'].nunique()} repos.")

#Fork data
def fetch_fork_info(repo_url: str, pr_number: int):
    """Return the fork repo URL for a given PR via GitHub API."""
    if "api.github.com/repos/" in repo_url:
        repo_path = repo_url.split("api.github.com/repos/")[-1].rstrip("/")
    else:
        repo_path = repo_url.replace("https://github.com/", "").rstrip(".git").rstrip("/")

    url = f"https://api.github.com/repos/{repo_path}/pulls/{pr_number}"

    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 403:
            print("Rate limited, sleeping 60s...")
            time.sleep(60)
            return None
        r.raise_for_status()
        data = r.json()
        head_repo = data.get("head", {}).get("repo", {})
        return head_repo.get("clone_url")
    except Exception as e:
        print(f"{repo_url}#{pr_number} failed: {e}")
        return None

#Fetch
results = []
for _, row in tqdm(pulls.iterrows(), total=len(pulls), desc="Fetching forks"):
    fork_url = fetch_fork_info(row["repo_url"], int(row["number"]))
    if not fork_url:
        continue
    results.append(fork_url)

# Deduplicate forks
forks = sorted(set(results))
print(f"Found {len(forks)} unique fork repos to clone.")

#Clone
for url in tqdm(forks, desc="Cloning forks"):
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = CLONE_DIR / name
    if dest.exists():
        print(f"Skipping existing repo: {name}")
        continue
    try:
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to clone {url}")

print("All fork repositories cloned successfully.")
