import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from decision_tree import DecisionTreeClassifier as MyDT
from random_forest import RandomForestClassifier as MyRF

# --- Quick sanity check on Iris ---
X_iris, y_iris = load_iris(return_X_y=True)
tree_iris = MyDT(max_depth=3)
tree_iris.fit(X_iris, y_iris)
iris_acc = (tree_iris.predict(X_iris) == y_iris).mean()
print(f"Iris dataset accuracy: {iris_acc:.4f}") 

# --- Load Wine Quality dataset ---
df = pd.read_csv('winequality-red.csv', sep=';')
X = df.drop('quality', axis=1).values
y = df['quality'].values
y = y - np.min(y)

# --- Checking missing values ---
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} columns")

# --- EXPERIMENT 1: Model Comparison (4 Approaches) ---
print("\n--- Running Experiment 1: Model Comparison ---")

models = {
    "My DT": MyDT(max_depth=10),
    "My RF": MyRF(n_estimators=50, max_depth=10),
    "Sklearn DT": SklearnDT(max_depth=10, random_state=42),
    "Sklearn RF": SklearnRF(n_estimators=50, max_depth=10, random_state=42)
}


exp1_results = []

for name, model in models.items():
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
 
    start_pred = time.time()
    test_preds = model.predict(X_test)
    pred_time = time.time() - start_pred
 
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (test_preds == y_test).mean()
 
    exp1_results.append({
        "Model": name,
        "Train Acc": train_acc,
        "Test Acc": test_acc,
        "Train Time": train_time,
        "Pred Time": pred_time
    })
    print(f"{name:<12} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Train Time: {train_time:.4f}s | Pred Time: {pred_time:.4f}s")
 
# Visualisation 1.1: Test Accuracy Comparison
res_df = pd.DataFrame(exp1_results)
plt.figure(figsize=(10, 5))
plt.bar(res_df['Model'], res_df['Test Acc'], color=['skyblue', 'salmon', 'lightgreen', 'orange'])
plt.title('Test Accuracy Comparison (Experiment 1)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('../figures/test_accuracy_comparison.png', dpi=300)
plt.show()

# Visualisation 1.2: Train vs Test Accuracy
x = np.arange(len(res_df))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, res_df['Train Acc'], width, label='Train Acc', color='steelblue')
ax.bar(x + width/2, res_df['Test Acc'], width, label='Test Acc', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(res_df['Model'])
ax.set_ylabel('Accuracy')
ax.set_title('Train vs Test Accuracy')
ax.legend()
plt.savefig('../figures/train_vs_test_accuracy.png', dpi=300)
plt.show()

# Visualisation 1.3: Training Time Comparison
plt.figure(figsize=(10, 5))
plt.bar(res_df['Model'], res_df['Train Time'], color='plum')
plt.yscale('log') 
plt.title('3. Training Time Comparison (Log Scale)')
plt.ylabel('Time (seconds) - Log Scale')
plt.savefig('../figures/time_comparison.png')
plt.show()

# --- EXPERIMENT 2: Hyperparameter Tuning ---
print("\n--- Running Experiment 2: Hyperparameter Tuning ---")
 
# --------------------------------------------------
# 2.1 Decision Tree: Grid Search (max_depth vs min_samples_split)
# --------------------------------------------------
depths = [1, 2, 3, 5, 10, 15, 20, None]
splits = [2, 5, 10, 20, 50]
heatmap_data = np.zeros((len(depths), len(splits)))
depth_labels = [str(d) if d is not None else 'None' for d in depths]

for i, d in enumerate(depths):
    for j, s in enumerate(splits):
        m = MyDT(max_depth=d, min_samples_split=s)
        m.fit(X_train, y_train)
        heatmap_data[i, j] = (m.predict(X_test) == y_test).mean()

# Visualisation 2.1: Grid search heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(heatmap_data, annot=True, fmt='.3f',
            xticklabels=splits, yticklabels=depth_labels, cmap='YlGnBu')
plt.title('DT Test Accuracy: max_depth vs min_samples_split')
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')
plt.savefig('../figures/dt_grid_search_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

best_i, best_j = np.unravel_index(heatmap_data.argmax(), heatmap_data.shape)
print(f"Best DT config: max_depth={depth_labels[best_i]}, "
      f"min_samples_split={splits[best_j]} → acc={heatmap_data[best_i, best_j]:.4f}")

# --------------------------------------------------
# 2.2 DT Learning curve: train/test vs max_depth
# (Bias-Variance analysis — required by assignment)
# --------------------------------------------------
print("\n[2.2] DT Bias-Variance: train/test accuracy vs max_depth")
depth_range = [1, 2, 3, 5, 7, 10, 15, 20, None]
dt_train_scores, dt_test_scores = [], []
 
for d in depth_range:
    m = MyDT(max_depth=d)
    m.fit(X_train, y_train)
    dt_train_scores.append((m.predict(X_train) == y_train).mean())
    dt_test_scores.append((m.predict(X_test) == y_test).mean())
 
# Visualisation 2.2: Bias-Variance analysis
x_labels = [str(d) if d is not None else 'None' for d in depth_range]
plt.figure(figsize=(10, 6))
plt.plot(x_labels, dt_train_scores, marker='o', label='Train Accuracy', color='blue')
plt.plot(x_labels, dt_test_scores,  marker='s', label='Test Accuracy',  color='red')
plt.fill_between(range(len(x_labels)),
                 dt_train_scores, dt_test_scores,
                 alpha=0.15, color='orange', label='Overfitting gap')
plt.xticks(range(len(x_labels)), x_labels)
plt.title('DT Bias-Variance: Accuracy vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('../figures/dt_bias_variance_depth.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# 2.3 RF Learning curve: n_estimators
# Required: [1, 5, 10, 25, 50, 100, 200]
# --------------------------------------------------
print("\n[2.3] RF Learning Curve: n_estimators")
n_trees_range = [1, 5, 10, 25, 50, 100, 200]
rf_train_scores, rf_test_scores = [], []
 
for n in n_trees_range:
    rf = MyRF(n_estimators=n, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_scores.append((rf.predict(X_train) == y_train).mean())
    rf_test_scores.append((rf.predict(X_test) == y_test).mean())
 
# Visualisation 2.3: n_estimators learning curve
plt.figure(figsize=(10, 6))
plt.plot(n_trees_range, rf_train_scores, marker='o', label='Train Accuracy', color='blue')
plt.plot(n_trees_range, rf_test_scores,  marker='s', label='Test Accuracy',  color='red')
plt.title('RF: Accuracy vs Number of Trees (n_estimators)')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('../figures/rf_n_estimators_learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# 2.4 RF max_features comparison
# Required: [1, 'sqrt', 'log2', None (all)]
# --------------------------------------------------
print("\n[2.4] RF max_features comparison")
mf_options = [1, 'sqrt', 'log2', None]
mf_labels  = ['1', 'sqrt', 'log2', 'all']
mf_train, mf_test = [], []
 
for mf in mf_options:
    rf = MyRF(n_estimators=50, max_depth=10, max_features=mf, random_state=42)
    rf.fit(X_train, y_train)
    mf_train.append((rf.predict(X_train) == y_train).mean())
    mf_test.append((rf.predict(X_test) == y_test).mean())
    print(f"  max_features={str(mf):<5} | Train: {mf_train[-1]:.4f} | Test: {mf_test[-1]:.4f}")
 
# Visualisation 2.4: max features comparison
x = np.arange(len(mf_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, mf_train, width, label='Train Acc', color='steelblue')
ax.bar(x + width/2, mf_test,  width, label='Test Acc',  color='coral')
ax.set_xticks(x)
ax.set_xticklabels(mf_labels)
ax.set_xlabel('max_features')
ax.set_ylabel('Accuracy')
ax.set_title('RF Accuracy vs max_features')
ax.legend()
plt.savefig('../figures/rf_max_features_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
 
# --------------------------------------------------
# 2.5 RF max_depth tuning
# Required: [5, 10, 20, None]
# --------------------------------------------------
print("\n[2.5] RF max_depth tuning")
rf_depths = [5, 10, 20, None]
rfd_train, rfd_test = [], []
 
for d in rf_depths:
    rf = MyRF(n_estimators=50, max_depth=d, random_state=42)
    rf.fit(X_train, y_train)
    rfd_train.append((rf.predict(X_train) == y_train).mean())
    rfd_test.append((rf.predict(X_test) == y_test).mean())
    print(f"  max_depth={str(d):<5} | Train: {rfd_train[-1]:.4f} | Test: {rfd_test[-1]:.4f}")
 
# Visualisation 2.5: max depth curve
d_labels = [str(d) if d is not None else 'None' for d in rf_depths]
plt.figure(figsize=(8, 5))
plt.plot(d_labels, rfd_train, marker='o', label='Train Accuracy', color='blue')
plt.plot(d_labels, rfd_test,  marker='s', label='Test Accuracy',  color='red')
plt.title('RF Accuracy vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('../figures/rf_max_depth_curve.png', dpi=300, bbox_inches='tight')
plt.show()
 
# --------------------------------------------------
# 2.6 Criterion comparison: Gini vs Entropy
# --------------------------------------------------
print("\n[2.6] Criterion comparison: Gini vs Entropy")
criteria = ['gini', 'entropy']
crit_results = []

for crit in criteria:
    m = MyDT(criterion=crit, max_depth=10)
    m.fit(X_train, y_train)
    acc = (m.predict(X_test) == y_test).mean()
    crit_results.append(acc)
    print(f"   criterion={crit:<8} | Test Acc: {acc:.4f}")

# Visualisation 2.6: Gini vs Entropy Comparison
plt.figure(figsize=(6, 5))
plt.bar(criteria, crit_results, color=['teal', 'coral'], width=0.5)
plt.title('Decision Tree: Gini vs Entropy Accuracy')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1.1)
for i, v in enumerate(crit_results):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

plt.savefig('../figures/criterion_comparison.png')
plt.show()

# ======================================================================
# EXPERIMENT 3: Feature Importance
# ======================================================================
print("\n--- Running Experiment 3: Feature Importance ---")
 
rf_best = models["My RF"]
importances = rf_best.get_feature_importance()
feature_names = df.drop('quality', axis=1).columns
sorted_idx = np.argsort(importances)[::-1]
 
print("\nAll features ranked by importance:")
for i in sorted_idx:
    print(f"  {feature_names[i]:<25} {importances[i]:.4f}")
 

# --------------------------------------------------
# 3․1։ Feature importance bar chart (all features)
# --------------------------------------------------
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], importances[sorted_idx], color='plum')
plt.xlabel('Importance Score')
plt.title('Feature Importances — My Random Forest')
plt.gca().invert_yaxis()
plt.savefig('../figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
 
# --------------------------------------------------
# 3․2։ Feature correlation heatmap
# --------------------------------------------------
plt.figure(figsize=(12, 9))
corr = pd.DataFrame(X_train, columns=feature_names).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig('../figures/feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
 
# --------------------------------------------------
# 3․3։ Performance vs number of top-k features
# Required k values: [1, 3, 5, 10] — using all 11 as the max
# --------------------------------------------------
print("\nPerformance vs top-k features:")
k_values = [1, 3, 5, 10, 11]
k_results = []
 
for k in k_values:
    top_k = sorted_idx[:k]
    m_k = MyRF(n_estimators=20, max_depth=10, random_state=42)
    m_k.fit(X_train[:, top_k], y_train)
    acc_k = (m_k.predict(X_test[:, top_k]) == y_test).mean()
    k_results.append(acc_k)
    print(f"  Top-{k:>2} features: {acc_k:.4f}")
 
plt.figure(figsize=(8, 5))
plt.plot(k_values, k_results, marker='D', color='red', linewidth=2)
plt.axhline(y=k_results[-1], color='gray', linestyle='--', label='All features baseline')
plt.title('Test Accuracy vs Number of Features Used')
plt.xlabel('Number of Features (k)')
plt.ylabel('Test Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.savefig('../figures/k_features_performance.png', dpi=300, bbox_inches='tight')
plt.show()
 
# --------------------------------------------------
# DT vs RF agreement on feature importance
# --------------------------------------------------
dt_model = MyDT(max_depth=10)
dt_model.fit(X_train, y_train)
dt_importances = dt_model.get_feature_importance()
dt_sorted = np.argsort(dt_importances)[::-1]
 
print("\nDT vs RF top-5 feature agreement:")
print(f"  {'Feature':<25} {'DT Rank':>8}  {'RF Rank':>8}")
for rank, (dt_i, rf_i) in enumerate(zip(dt_sorted[:5], sorted_idx[:5]), 1):
    print(f"  DT#{rank}: {feature_names[dt_i]:<22} | RF#{rank}: {feature_names[rf_i]}")
 

# --------------------------------------------------
# Training Time vs Number of Trees
# --------------------------------------------------
print("\nChecking Computational Complexity...")
n_trees_range = [1, 10, 50, 100]
times = []
for n in n_trees_range:
    start = time.time()
    m = MyRF(n_estimators=n, max_depth=10, random_state=42)
    m.fit(X_train, y_train)
    times.append(time.time() - start)

plt.figure(figsize=(8, 5))
plt.plot(n_trees_range, times, marker='o', color='green')
plt.title('Training Time vs Number of Trees (RF)')
plt.xlabel('Number of Trees')
plt.ylabel('Time (seconds)')
plt.savefig('../figures/complexity_n_trees.png')
plt.show()

# ======================================================================
# Verify predict_proba
# ======================================================================
print("\n--- predict_proba sanity check ---")
proba = rf_best.predict_proba(X_test[:3])
print("Probabilities (first 3 test samples):")
print(proba)
print("Row sums (should all be 1.0):", proba.sum(axis=1))
 
print("\nAll experiments completed successfully!")
 