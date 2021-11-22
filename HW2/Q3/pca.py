import numpy as np
import pandas 

# pca with fit transform
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.mean = None
        self.components = None

    def fit(self, X):
        X = X.copy()
        self.mean = X.mean(axis=0)
        X -= self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        self.components = self.eigenvectors[:, :self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return X.dot(self.components)

    def explained_variance_ratio(self):
        return self.eigenvalues / self.eigenvalues.sum()


