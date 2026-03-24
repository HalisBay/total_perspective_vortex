from ml_service.machine_learning.data_processor import DataProcessor
# from ml_service.machine_learning.feature_extractor import FeatureExtractor
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

processor = DataProcessor()
# extractor = FeatureExtractor()


class TrainingPipeline:
    def __init__(self):
        self.pipeline = None

    def get_train_data(self, epochs):
        X = epochs.get_data()
        y = epochs.events[:, -1]

        y = np.where(y == 2, 0, 1)

        return X, y

    def create_csp_lda_pipeline(self):
        csp = CSP()
        lda = LinearDiscriminantAnalysis()
        self.pipeline = Pipeline([("csp", csp), ("lda", lda)])

        return self.pipeline

    def get_cv_score(self, X, y):
        if self.pipeline is None:
            self.create_csp_lda_pipeline()
        scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="accuracy")
        print(f"scores: {[f'{s:.2f}' for s in scores]}")
        print(f"mean acc: {scores.mean():.2f}")

    def fit(self, X, y):
        if self.pipeline is None:
            self.create_csp_lda_pipeline()

        self.pipeline.fit(X, y)
        print(f"Model fitted")

    def predict(self, X):
        return self.pipeline.predict(X)
