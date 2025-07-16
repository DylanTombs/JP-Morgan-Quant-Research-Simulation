import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./Data/Task 3 and 4_Loan_Data.csv")

features = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score"
]
target = "default"

### --- Decision Tree ---
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini(y):
    if len(y) == 0:
        return 0
    p = np.mean(y)
    return 1 - p**2 - (1 - p)**2

def bestSplit(X, y, features):
    bestGini = float('inf')
    bestFeature, bestThreshold = None, None

    for i, feature in enumerate(features):
        thresholds = np.unique(X[:, i])
        for t in thresholds:
            leftMask = X[:, i] <= t
            rightMask = ~leftMask

            gLeft = gini(y[leftMask])
            gRight = gini(y[rightMask])
            gTotal = (len(y[leftMask]) * gLeft + len(y[rightMask]) * gRight) / len(y)

            if gTotal < bestGini:
                bestGini = gTotal
                bestFeature = i
                bestThreshold = t

    return bestFeature, bestThreshold

def buildTree(X, y, features, depth=0, maxDepth=4, minSamples=5):
    if depth >= maxDepth or len(y) < minSamples or len(np.unique(y)) == 1:
        return TreeNode(value=np.mean(y))

    featureIdx, threshold = bestSplit(X, y, features)
    if featureIdx is None:
        return TreeNode(value=np.mean(y))

    leftMask = X[:, featureIdx] <= threshold
    rightMask = ~leftMask

    left = buildTree(X[leftMask], y[leftMask], features, depth + 1, maxDepth, minSamples)
    right = buildTree(X[rightMask], y[rightMask], features, depth + 1, maxDepth, minSamples)

    return TreeNode(feature=featureIdx, threshold=threshold, left=left, right=right)

def predictTree(tree, x):
    node = tree
    while node.value is None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

### --- Train/Test Split ---
X = df[features].values
y = df[target].values.reshape(-1, 1)

np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
trainIdx, testIdx = indices[:split], indices[split:]

XTrain, XTest = X[trainIdx], X[testIdx]
yTrain, yTest = y[trainIdx], y[testIdx]

means = XTrain.mean(axis=0)
stds = XTrain.std(axis=0)

XTrainNorm = (XTrain - means) / stds
XTestNorm = (XTest - means) / stds

XTrainBias = np.hstack([np.ones((XTrain.shape[0], 1)), XTrainNorm])
XTestBias = np.hstack([np.ones((XTest.shape[0], 1)), XTestNorm])

### --- Logistic Regression ---
weights = np.zeros((XTrainBias.shape[1], 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def trainLogistic(X, y, weights, lr=0.01, epochs=10000):
    for _ in range(epochs):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        gradient = np.dot(X.T, predictions - y) / len(y)
        weights -= lr * gradient
    return weights

weights = trainLogistic(XTrainBias, yTrain, weights)

tree = buildTree(XTrain, yTrain.flatten(), features)

### --- Evaluation ---
def accuracy(yTrue, yPred, threshold=0.5):
    preds = np.array(yPred) >= threshold
    return np.mean(preds.flatten() == yTrue.flatten())

def logLoss(yTrue, yPred):
    eps = 1e-10
    yPred = np.clip(yPred, eps, 1 - eps)
    return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))

logisticPreds = sigmoid(np.dot(XTestBias, weights)).flatten()
treePreds = [predictTree(tree, x) for x in XTest]

logAcc = accuracy(yTest, logisticPreds)
treeAcc = accuracy(yTest, treePreds)

logLL = logLoss(yTest, logisticPreds)
treeLL = logLoss(yTest, treePreds)

print("--- Logistic Regression ---")
print(f"Accuracy: {logAcc:.3f}")
print(f"Log Loss: {logLL:.4f}\n")

print("--- Decision Tree ---")
print(f"Accuracy: {treeAcc:.3f}")
print(f"Log Loss: {treeLL:.4f}")

### --- Expected Loss  ---
def predictExpectedLoss(borrowerDict):
    x = np.array([borrowerDict[feat] for feat in features])
    xNorm = (x - means) / stds
    xInput = np.concatenate(([1], xNorm))
    z = np.dot(xInput, weights)
    pdEstimate = sigmoid(z)[0]
    expectedLoss = pdEstimate * 0.9 * borrowerDict["loan_amt_outstanding"]
    return {"PD": pdEstimate, "Expected Loss": expectedLoss}

def predictExpectedLossTree(borrowerDict):
    x = np.array([borrowerDict[feat] for feat in features])
    pdEst = predictTree(tree, x)
    expectedLoss = pdEst * 0.9 * borrowerDict["loan_amt_outstanding"]
    return {"PD": pdEst, "Expected Loss": expectedLoss}

### --- Example Borrower ---
borrower = {
    "credit_lines_outstanding": 2,
    "loan_amt_outstanding": 4000,
    "total_debt_outstanding": 10000,
    "income": 60000,
    "years_employed": 3,
    "fico_score": 650
}

print("\n--- Borrower Predictions ---")
resLog = predictExpectedLoss(borrower)
resTree = predictExpectedLossTree(borrower)

print(f"Logistic Regression PD: {resLog['PD']:.3f}, Expected Loss: ${resLog['Expected Loss']:.2f}")
print(f"Decision Tree       PD: {resTree['PD']:.3f}, Expected Loss: ${resTree['Expected Loss']:.2f}")

### --- Plot PD Distribution ---
plt.hist(logisticPreds, bins=15, alpha=0.6, label="Logistic")
plt.hist(treePreds, bins=15, alpha=0.6, label="Decision Tree")
plt.xlabel("Predicted PD")
plt.ylabel("Frequency")
plt.legend()
plt.title("PD Distribution on Test Set")
plt.show()

