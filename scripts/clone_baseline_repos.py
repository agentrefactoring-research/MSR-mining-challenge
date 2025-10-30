import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

REPO_LIST = Path("data/processed/java_baseline_repos.csv")
OUT_DIR = Path("repos_baseline")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(REPO_LIST)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Cloning repos"):
    url, name = row["repo_url"], row["name"]
    dest = OUT_DIR / name
    if dest.exists():
        print(f"⏭️  Skipping existing repo: {name}")
        continue
    try:
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", url, str(dest)], check=True)
        print(f"✅ Cloned {name}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to clone {name} ({url})")
