import numpy as np
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest ensemble classifier.
    Reduces variance over single trees through bagging and feature randomness.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        """
        Args:
            n_estimators    : Number of trees in the forest
            max_depth       : Maximum depth of each tree (None = unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf : Minimum samples required at each leaf
            max_features    : Features considered at each split —
                              'sqrt', 'log2', int, or None (all)
            random_state    : Seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.n_classes_ = None

    def _bootstrap_sample(self, X, y):
        """
        Draw a bootstrap sample (sampling with replacement).
        Each tree sees a different random subset of the training data.

        Returns:
            tuple: (X_sample, y_sample) with the same shape as inputs
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        """
        Train an ensemble of decision trees using bootstrap sampling.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Integer class labels starting from 0 (n_samples,)

        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_classes_ = len(np.unique(y))
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Predict class labels by majority voting across all trees.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            numpy.ndarray: Predicted class labels (n_samples,)
        """
        # Shape: (n_samples, n_estimators)
        tree_preds = np.array([tree.predict(X) for tree in self.trees]).T
        return np.array([
            np.bincount(row, minlength=self.n_classes_).argmax()
            for row in tree_preds
        ])

    def predict_proba(self, X):
        """
        Estimate class probabilities as the proportion of votes per class.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            numpy.ndarray: Class probabilities (n_samples, n_classes)
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees]).T
        return np.array([
            np.bincount(row, minlength=self.n_classes_) / self.n_estimators
            for row in tree_preds
        ])

    def get_feature_importance(self):
        """
        Return average normalised feature importances across all trees.

        Returns:
            numpy.ndarray: Feature importance scores (n_features,)
        """
        if not self.trees:
            raise ValueError("Model must be fitted before getting feature importances.")
        all_importances = np.array([tree.get_feature_importance() for tree in self.trees])
        return np.mean(all_importances, axis=0)