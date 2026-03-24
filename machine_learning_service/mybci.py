from ml_service.data_layer.data_connector import load_subdataset
from ml_service.machine_learning.data_processor import DataProcessor
from ml_service.machine_learning.training_pipeline import TrainingPipeline
from ml_service.data_layer.object_connector import ObjectConnector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

processor = DataProcessor()
trainer = TrainingPipeline()
oconnector = ObjectConnector()

if __name__ == "__main__":
    raws = load_subdataset()
    raw = raws[0]  # subject
    filter_raw = processor.clean_eeg_data(raw, [0,5,9,16]) #  2, 4, 8, 10

    events, e_id = processor.find_events(filter_raw)
    epochs = processor.create_epochs(filter_raw, events, e_id)
    X, y = trainer.get_train_data(epochs=epochs)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Class distribution: {np.bincount(y)}")

    trainer.get_cv_score(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    trainer.fit(X_train, y_train)
    oconnector.save_model(trainer.pipeline,"csplda.joblib")

    model = oconnector.load_model("csplda.joblib")
    oconnector.get_model_info(model)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"acc : {acc:.2f}")
