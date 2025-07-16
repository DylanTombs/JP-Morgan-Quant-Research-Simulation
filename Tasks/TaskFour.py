import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./Data/Task 3 and 4_Loan_Data.csv")
ficoScores = df["fico_score"].values
defaults = df["default"].values

sortedIdx = np.argsort(ficoScores)
ficoSorted = ficoScores[sortedIdx]
n = len(ficoSorted)
k = 10 

cumsum = np.cumsum(ficoSorted)
cumsumSq = np.cumsum(ficoSorted**2)

def segmentMSE(i, j):
    count = j - i + 1
    if count == 0:
        return 0
    sum_ = cumsum[j] - (cumsum[i - 1] if i > 0 else 0)
    sumSq = cumsumSq[j] - (cumsumSq[i - 1] if i > 0 else 0)
    mean = sum_ / count
    return sumSq - 2 * mean * sum_ + count * mean**2  

dp = np.full((n, k), np.inf)
boundaries = np.full((n, k), -1)
for i in range(n):
    dp[i][0] = segmentMSE(0, i)

for buckets in range(1, k):
    for i in range(buckets, n):
        for j in range(buckets - 1, i):
            cost = dp[j][buckets - 1] + segmentMSE(j + 1, i)
            if cost < dp[i][buckets]:
                dp[i][buckets] = cost
                boundaries[i][buckets] = j

idx = n - 1
bucketEnds = []
for b in reversed(range(k)):
    bucketEnds.append(idx)
    idx = boundaries[idx][b]
bucketEnds.reverse()

bucketLabels = [f"Rating {i+1}" for i in range(k)]
bucketScores = [ficoSorted[i] for i in bucketEnds]

def assignMSEBucket(score):
    for i, val in enumerate(bucketScores):
        if score <= val:
            return (f"Rating {i + 1}", i + 1)
    return (f"Rating {k}", k)

# Apply and split into two columns
df[["fico_rating_mse", "fico_rating_num_mse"]] = df["fico_score"].apply(assignMSEBucket).apply(pd.Series)

mseStats = df.groupby("fico_rating_mse")["default"].agg(["count", "sum"])
mseStats["PD"] = mseStats["sum"] / mseStats["count"]
mseStats["rating_num"] = df.groupby("fico_rating_mse")["fico_rating_num_mse"].first()

mseStatsSorted = mseStats.sort_values("rating_num")

plt.figure(figsize=(8, 5))
plt.bar(mseStatsSorted.index, mseStatsSorted["PD"], color="skyblue")
plt.title("Default Rate by FICO Rating (MSE Buckets)")
plt.ylabel("Probability of Default")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nMSE Quantization Summary")
print(mseStatsSorted.head)

