from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

"""
CSP, iki sınıfı ayırmak için uzaysal filtreler bulur.

1. Amaç:
- Sınıf 1 için varyansı büyük olan yönleri bul
- Aynı yönlerde Sınıf 2’nin varyansı küçük olsun
- Böylece bu yönlerdeki güç değerleri sınıflandırıcı için ayırt edici hale gelir

2. Kovaryans matrisleri:
Her sınıf için kovaryans: C1, C2
Ortak kovaryans: C_sum = C1 + C2

3. Whitening:
Ortak kovaryans C_sum kullanılarak whitening transform uygulanır
Böylece ortak uzaya dönüşüm yapılır( sınıf farkı daha net görünür)

4. Eigendecomposition:
C1 üzerinde eigendecomposition yapılır
Elde edilen özvektörler CSP filtreleri olur



"""


class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=4,
        log=None,
        component_order="mutual_info",
    ):
        self.n_components = n_components
        self.log = log
        self.component_order = component_order
        self.eps = 1e-9
        self.filters = None

    def fit(self, X, y):
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array but {X.ndim}D")

        b_classes = np.unique(y)
        if len(b_classes) != 2:
            raise ValueError(f"CSP need binary structure")

        # verinin dağılımı
        cov1, cov2 = self.calculate_covariance(X, y, b_classes)
        cov_total = cov1 + cov2
        # verinin hangi yönde ne kadar dağıldığı
        eig_vals, eig_vecs = np.linalg.eigh(cov_total)
        # taşmaya yol açabilir diye eps ekledim.
        eig_vals = np.maximum(eig_vals, self.eps)
        # veriyi tüm yönleri eşit biçimde dönüştür
        whiten_cov_matx = np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
        # cov 1 normalize edilmiş uzaya taşı
        S1 = whiten_cov_matx @ cov1 @ whiten_cov_matx.T
        # cov 1 i hangi yön büyütüyor
        eig_vals, eig_vecs = np.linalg.eigh(S1)

        # Not: pca gibi sadece en büyük özdeğerleri başa almak yerine,
        # MNE gibi farklı sıralama ve normalizasyon işlemleri uyguluyorum.

        if self.component_order == "mutual_info":
            # 0.5ten uzak olan değerler en başa gelsin.
            idx = np.argsort(np.abs(eig_vals - 0.5))[::-1]
        elif self.component_order == "alternate":
            # küçük ve büyük  değerleri sırayla karıştır
            i = np.argsort(eig_vals)
            idx = np.empty_like(i)
            idx[1::2] = i[: len(i) // 2]
            idx[0::2] = i[len(i) // 2 :][::-1]
        else:
            raise ValueError("component_order must be 'mutual_info' or 'alternate'")

        eig_vecs = eig_vecs[:, idx]

        W = eig_vecs.T @ whiten_cov_matx

        self.filters = W[: self.n_components]

        return self

    def transform(self, X):
        if self.filters is None:
            raise ValueError("Model has not been fitted")
        n_samples = X.shape[0]
        n_filters = self.filters.shape[0]

        X_csp = np.zeros((n_samples, n_filters))

        for i, sample in enumerate(X):
            csp_fiter = self.filters @ sample
            power = (csp_fiter**2).mean(axis=1)
            if self.log:
                X_csp[i, :] = np.log(np.maximum(power, self.eps))
            else:
                X_csp[i, :] = power
        return X_csp

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def calculate_covariance(self, X, y, b_classes):
        n_samples, n_channels, n_times = X.shape
        result_matrix = []

        for cl in b_classes:
            x_cl = X[y == cl]

            cov_mat = np.zeros((n_channels, n_channels))

            for epoch in x_cl:
                epoch_cov = epoch @ epoch.T
                epoch_cov /= np.trace(epoch_cov)
                cov_mat += epoch_cov

            cov_mat /= len(x_cl)
            result_matrix.append(cov_mat)

        return result_matrix
