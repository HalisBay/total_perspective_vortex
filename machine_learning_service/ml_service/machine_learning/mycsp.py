from sklearn.base import BaseEstimator, TransformerMixin

"""
CSPnin amacı: iki sınıfı ayıran en iyi projeksiyon yönlerini bulmak

covariance hesabı
eigen decomposition nasıl yapılacak ?
sonuç to feature ?

"""


class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=4,
        reg=None,
        log=True,
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log

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
        # veriyi tüm yönleri eşit biçimde dönüştür
        whiten_cov_matx = np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
        # cov 1 normalize edilmiş uzaya taşı
        S1 = whiten_cov_matx @ cov1 @ whiten_cov_matx.T
        # cov 1 i hangi yön büyütüyor
        eig_vals, eig_vecs = np.linalg.eigh(S1)
        # en önemli yönler başa gelsin diye sırala
        ix = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, ix]

        W = eig_vecs.T @ whiten_cov_matx

        self.filters_ = W[: self.n_components]

        return self

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
