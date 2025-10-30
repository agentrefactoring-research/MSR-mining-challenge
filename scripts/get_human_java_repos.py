import os
import csv
import random
import requests
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "java_baseline_repos.csv"

MIN_STARS = 50
MAX_REPOS = 100
SELECTED_REPOS = 86
PUSHED_BEFORE = "2021-01-01"

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise EnvironmentError("Please set GITHUB_TOKEN in your environment.")

HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"token {TOKEN}"

#Query GitHUb
def get_human_written_java_repos(min_stars=50, pushed_before="2021-01-01", max_repos=500, max_pages=20):
    """Fetch Java repositories last pushed before a given date (likely human-written)."""
    repos = []
    per_page = 100
    for page in range(1, max_pages + 1):
        url = (
            f"https://api.github.com/search/repositories?"
            f"q=language:Java+stars:>={min_stars}+pushed:<{pushed_before}"
            f"&sort=stars&order=desc&per_page={per_page}&page={page}"
        )
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()

        items = r.json().get("items", [])
        if not items:
            break

        for item in items:
            repo_url = item["html_url"] + ".git"
            name = item["name"]
            size_gb = item["size"] / 1_000_000
            repos.append({
                "repo_url": repo_url,
                "name": name,
                "size_gb": round(size_gb, 9)
            })

            if len(repos) >= max_repos:
                return repos
    return repos

def save_to_csv(repos, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["repo_url", "name", "size_gb"])
        writer.writeheader()
        writer.writerows(repos)
    print(f"Saved → {output_path}")

def main():
    print("Fetching Java repositories from GitHub...")
    repos = get_human_written_java_repos(
        min_stars=MIN_STARS,
        pushed_before=PUSHED_BEFORE,
        max_repos=MAX_REPOS
    )

    print(f"Retrieved {len(repos)} repositories.")
    print(f"Randomly selecting {SELECTED_REPOS} of them...")

    random.shuffle(repos)
    selected = repos[:SELECTED_REPOS]

    save_to_csv(selected, OUTPUT_CSV)
    print(f"🏁 Done. {SELECTED_REPOS} random repos saved to {OUTPUT_CSV.name}")

if __name__ == "__main__":
    main()
