import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

#Load data
df = pd.read_csv("G:\msr-mining-challenge-2026\data\processed\smell_deltas_per_commit.csv")

#Separate human and agentic data
human = df[df["agent"] == "Human"]["delta"].dropna()

#Get unique agent names (excluding Human)
agents = df[df["agent"] != "Human"]["agent"].unique()

#Function to compute Cliff's delta
def cliffs_delta(x, y):
    """Compute Cliff's delta for two independent samples."""
    n1, n2 = len(x), len(y)
    # Pairwise comparisons
    greater = sum(a > b for a in x for b in y)
    less = sum(a < b for a in x for b in y)
    delta = (greater - less) / (n1 * n2)
    return delta

#Function to interpret Cliff's delta
def interpret_delta(delta):
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"

#Run tests for each agent vs human
results = []
for agent in agents:
    agent_data = df[df["agent"] == agent]["delta"].dropna()

#Mann–Whitney U test
    stat, p_value = mannwhitneyu(human, agent_data, alternative='two-sided')

#Cliff's delta
    delta = cliffs_delta(agent_data.values, human.values)
    interpretation = interpret_delta(delta)

    results.append({
        "Agent": agent,
        "U-statistic": stat,
        "p-value": p_value,
        "Cliffs_delta": delta,
        "Effect_size": interpretation,
        "Human_median": human.median(),
        f"{agent}_median": agent_data.median(),
        "Human_mean": human.mean(),
        f"{agent}_mean": agent_data.mean(),
        "nHuman": len(human),
        f"n{agent}": len(agent_data)
    })

#Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)