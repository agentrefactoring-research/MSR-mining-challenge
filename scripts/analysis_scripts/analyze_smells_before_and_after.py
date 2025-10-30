import subprocess
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import shutil
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
TEMP_DIR = DATA_DIR / "designite_temp"
DESIGNITE_JAR = PROJECT_ROOT / "tools" / "DesigniteJava.jar"

AGENTIC_COMMITS = DATA_DIR / "agentic_refactoring_commits.parquet"
HUMAN_COMMITS = DATA_DIR / "baseline_refactoring_commits.parquet"
REPOS_AGENTIC = PROJECT_ROOT / "repos_forks"
REPOS_HUMAN = PROJECT_ROOT / "repos_baseline"

for d in [DATA_DIR, TABLES_DIR, LOGS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "designite_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
)
print(f"Logging to {LOG_FILE}")


def run_subprocess(cmd, timeout=300):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return result.returncode == 0, result.stdout.decode(errors="ignore"), result.stderr.decode(errors="ignore")
    except subprocess.TimeoutExpired:
        return False, "", "Timeout expired"
    except Exception as e:
        return False, "", str(e)

def ensure_repo(repo_path: Path, full_name: str, dataset: str):
    if not repo_path.exists():
        url = f"https://github.com/{full_name}.git"
        logging.warning(f"Repo not found for {full_name}. Cloning...")
        ok, _, err = run_subprocess(["git", "clone", url, str(repo_path)], timeout=600)
        if not ok:
            logging.error(f"Failed to clone {url}: {err}")
            return False
    else:
        run_subprocess(["git", "-C", str(repo_path), "fetch", "--all"])
    return True

def get_changed_files(repo: Path, sha: str) -> list[str]:
    ok, out, err = run_subprocess(
        ["git", "-C", str(repo), "diff-tree", "--no-commit-id", "--name-only", "-r", sha]
    )
    if not ok:
        logging.warning(f"git diff-tree failed for {repo}@{sha}: {err}")
        return []
    return [f for f in out.splitlines() if f.strip().endswith(".java")]

def checkout_commit(repo: Path, sha: str) -> bool:
    ok, _, err = run_subprocess(["git", "-C", str(repo), "checkout", "-f", sha])
    if not ok:
        logging.error(f"❌ Git checkout failed for {repo}@{sha}: {err}")
    return ok

def copy_subset(repo: Path, changed_files: list[str]) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="subset_", dir=TEMP_DIR))
    copied = 0
    for fpath in changed_files:
        src = repo / fpath
        if not src.exists():
            continue
        dest = temp_dir / fpath
        try:
            if len(str(dest)) > 240:
                flat_name = "_".join(fpath.replace("\\", "/").split("/")[-5:])
                dest = temp_dir / flat_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
        except Exception as e:
            logging.warning(f"⚠️ Could not copy {src}: {e}")
    return temp_dir

def run_designite(input_dir: Path, output_dir: Path, label: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java", "-Xmx6G", "-jar", str(DESIGNITE_JAR),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", "-f", "csv"
    ]
    logging.info(f"Running Designite on {label}")
    ok, out, err = run_subprocess(cmd, timeout=900)
    if not ok:
        logging.error(f"❌ Designite failed for {label}")
        logging.error(err[:300])
        return 0
    return count_smells(output_dir)

def count_smells(output_dir: Path) -> int:
    total = 0
    for csv in output_dir.glob("*.csv"):
        if "Metric" in csv.name or "Summary" in csv.name:
            continue
        try:
            df = pd.read_csv(csv)
            total += len(df)
        except Exception as e:
            logging.warning(f"Could not read {csv.name}: {e}")
    return total


print("Loading commit datasets...")
agentic = pd.read_parquet(AGENTIC_COMMITS)
human = pd.read_parquet(HUMAN_COMMITS)
agentic["dataset"], human["dataset"] = "Agentic", "Human"
combined = pd.concat([agentic, human], ignore_index=True)
combined = combined[combined["has_refactoring"] == True]
print(f"✅ Loaded {len(combined)} refactoring commits across datasets.")

results = []
start_time = time.time()

for i, row in tqdm(combined.iterrows(), total=len(combined), desc="Analyzing commits"):
    repo_name = row["full_name"].split("/")[-1]
    full_name, sha = row["full_name"], row["sha"]
    dataset, agent = row["dataset"], row["agent"]
    repo = (REPOS_AGENTIC if dataset == "Agentic" else REPOS_HUMAN) / repo_name

    if not ensure_repo(repo, full_name, dataset):
        continue

    changed = get_changed_files(repo, sha)
    num_changed = len(changed)
    logging.info(f"{dataset}/{agent}/{repo_name}@{sha[:8]}: {num_changed} files changed")

    if num_changed == 0:
        logging.info(f"Skipping {repo_name}@{sha[:8]} — 0 files changed")
        continue

    label = f"{dataset}/{agent}/{repo_name}@{sha[:8]}"
    t0 = time.time()

    #Before refactor files
    if checkout_commit(repo, f"{sha}^"):
        subset_before = copy_subset(repo, changed)
        smells_before = run_designite(subset_before, TEMP_DIR / f"{repo_name}_{sha[:8]}_before", f"{label}_before")
        shutil.rmtree(subset_before, ignore_errors=True)
    else:
        smells_before = 0

    #After refactor files
    if checkout_commit(repo, sha):
        subset_after = copy_subset(repo, changed)
        smells_after = run_designite(subset_after, TEMP_DIR / f"{repo_name}_{sha[:8]}_after", f"{label}_after")
        shutil.rmtree(subset_after, ignore_errors=True)
    else:
        smells_after = 0

    delta = smells_after - smells_before
    elapsed = time.time() - t0

    results.append({
        "dataset": dataset, "agent": agent, "repo": repo_name,
        "commit": sha, "smells_before": smells_before,
        "smells_after": smells_after, "delta": delta,
        "runtime_sec": round(elapsed, 2)
    })

    print(f"{label}: Δ={delta}, before={smells_before}, after={smells_after}, {elapsed:.1f}s")

#Output
df = pd.DataFrame(results)
out_csv = DATA_DIR / "smell_deltas_per_commit.csv"
df.to_csv(out_csv, index=False)
print(f"💾 Saved → {out_csv}")

if not df.empty:
    summary = (
        df.groupby(["dataset", "agent"])[["smells_before", "smells_after", "delta"]]
        .agg(["mean", "median", "std", "min", "max"]).round(2)
    )
    summary.to_csv(TABLES_DIR / "smell_summary_stats_by_agent.csv")
    print("Summary saved.")
else:
    print("No valid results.")

print(f"✅ Done in {(time.time()-start_time)/60:.2f} min total.")
