import numpy as np

class TreeNode:
    """
    A node in the decision tree.
    Stores split information or the final class label if it's a leaf.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, 
                 is_leaf=False, value=None, samples=None, impurity=None):
        self.feature = feature      # Index of feature used for split
        self.threshold = threshold    # Threshold value for binary split
        self.left = left              # Left child node (<= threshold)
        self.right = right            # Right child node (> threshold)
        self.is_leaf = is_leaf        # True if node is a leaf
        self.value = value            # Class label (only for leaf nodes)
        self.samples = samples        # Number of samples in node
        self.impurity = impurity      # Gini/Entropy value

class DecisionTreeClassifier:
    """
    CART Decision Tree for classification.
    Supports Gini impurity and Entropy, binary splits on numerical features.
    """
 
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None):
        """
        Args:
            criterion       : 'gini' or 'entropy'
            max_depth       : Maximum tree depth (None = unlimited)
            min_samples_split: Minimum samples needed to attempt a split
            min_samples_leaf : Minimum samples that must remain in each leaf
            max_features    : Features to consider at each split —
                              None (all), 'sqrt', 'log2', or int
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None

    # ------------------------------------------------------------------
    # Impurity calculations
    # ------------------------------------------------------------------

    def _gini(self, y):
        """
        Calculate Gini impurity.
        Gini(D) = 1 - sum(p_k^2)
        """
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - np.sum(p ** 2)
 
    def _entropy(self, y):
        """
        Calculate Shannon Entropy.
        H(D) = -sum(p_k * log2(p_k))
        """
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        p = p[p > 0]   # avoid log(0)
        return -np.sum(p * np.log2(p))
 
    def _impurity(self, y):
        """Dispatch to the correct criterion."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        return self._gini(y)

    # ------------------------------------------------------------------
    # Splitting helpers
    # ------------------------------------------------------------------
 
    def _split(self, X, feature_idx, threshold):
        """
        Split samples into left (<= threshold) and right (> threshold).
 
        Returns:
            tuple: (left_indices, right_indices)
        """
        left_idx = np.where(X[:, feature_idx] <= threshold)[0]
        right_idx = np.where(X[:, feature_idx] > threshold)[0]
        return left_idx, right_idx
 

    def _information_gain(self, y, left_idx, right_idx):
        """
        IG = Impurity(parent) - weighted_avg(Impurity(children))
        """
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0.0
        n, n_l, n_r = len(y), len(left_idx), len(right_idx)
        child_impurity = (n_l / n) * self._impurity(y[left_idx]) + \
                         (n_r / n) * self._impurity(y[right_idx])
        return self._impurity(y) - child_impurity
 
    def _get_feature_indices(self, n_features):
        """
        Return the subset of feature indices to consider for this split.
        Supports None (all), 'sqrt', 'log2', or an integer count.
        """
        if self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        else:
            return np.arange(n_features)   # None → all features
        return np.random.choice(n_features, k, replace=False)

    def _find_best_split(self, X, y):
        """
        Find the optimal feature and threshold to split the data.
        Iterates over selected features and all unique thresholds,
        choosing the split that maximises information gain.
 
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
 
        Returns:
            tuple: (best_feature_index, best_threshold, best_gain)
            Returns (None, None, 0) if no valid split is found.
 
        Notes:
            - Uses greedy search (considers all candidate splits).
            - Time complexity: O(n_features x n_samples x log n_samples).
            - Only binary splits are considered.
        """
        best_gain = -1.0
        split_idx, split_thresh = None, None
 
        for feat_idx in self._get_feature_indices(X.shape[1]):
            for threshold in np.unique(X[:, feat_idx]):
                l_idx, r_idx = self._split(X, feat_idx, threshold)
 
                # Enforce min_samples_leaf on both children
                if len(l_idx) < self.min_samples_leaf or \
                   len(r_idx) < self.min_samples_leaf:
                    continue
 
                gain = self._information_gain(y, l_idx, r_idx)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, threshold
 
        return split_idx, split_thresh, best_gain
    
    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
 
        Stopping conditions:
            - Only one unique class remains
            - Too few samples to split (min_samples_split)
            - Maximum depth reached
            - No split yields positive information gain
        """
        n_samples = X.shape[0]
 
        # --- Stopping conditions ---
        if (len(np.unique(y)) == 1
                or n_samples < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return TreeNode(is_leaf=True,
                            value=self._most_common_label(y),
                            samples=n_samples,
                            impurity=self._impurity(y))
 
        f_idx, thresh, gain = self._find_best_split(X, y)
 
        # No beneficial split found
        if f_idx is None or gain <= 0:
            return TreeNode(is_leaf=True,
                            value=self._most_common_label(y),
                            samples=n_samples,
                            impurity=self._impurity(y))
 
        # Accumulate weighted feature importance (sklearn convention)
        self.feature_importances_[f_idx] += gain * n_samples
 
        l_idx, r_idx = self._split(X, f_idx, thresh)
        left = self._build_tree(X[l_idx], y[l_idx], depth + 1)
        right = self._build_tree(X[r_idx], y[r_idx], depth + 1)
 
        return TreeNode(f_idx, thresh, left, right,
                        samples=n_samples,
                        impurity=self._impurity(y))
 
    
    def fit(self, X, y):
        """
        Train the decision tree.
 
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Integer class labels starting from 0 (n_samples,)
 
        Returns:
            self
        """
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._build_tree(X, y)
        return self

    def _traverse_tree(self, x, node):
        """Traverse the tree for a single sample."""
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
 

    def predict(self, X):
        """
        Predict class labels for samples in X.
 
        Args:
            X: Feature matrix (n_samples, n_features)
 
        Returns:
            numpy.ndarray: Predicted class labels (n_samples,)
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _most_common_label(self, y):
        """
        Return the most frequent class label.
        Used to set the value of leaf nodes.
 
        Args:
            y: Array of integer class labels
 
        Returns:
            int | None: Most frequent label, or None if y is empty.
        """
        if len(y) == 0:
            return None
        return int(np.bincount(y).argmax())

    def get_feature_importance(self):
        """
        Return normalised feature importances based on weighted information gain.
 
        Returns:
            numpy.ndarray: Importance scores summing to 1 (n_features,)
 
        Notes:
            - Importance is accumulated during fit() as gain * n_samples.
            - Higher values indicate more predictive features.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted before getting feature importances.")
        total = np.sum(self.feature_importances_)
        if total > 0:
            return self.feature_importances_ / total
        return self.feature_importances_
 