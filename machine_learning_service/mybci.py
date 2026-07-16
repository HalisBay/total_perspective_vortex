from ml_service.data_layer.data_connector import load_subdataset
from ml_service.machine_learning.data_processor import DataProcessor
from ml_service.machine_learning.training_pipeline import TrainingPipeline
from ml_service.data_layer.object_connector import ObjectConnector
from ml_service.applications.visualize import plotter, filter_plotter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import time
import argparse
import mne

processor = DataProcessor()
trainer = TrainingPipeline()
oconnector = ObjectConnector()


def load_data(subjects, runs):
    raws = load_subdataset(subjects, runs)
    if len(raws) > 1:
        raw = mne.concatenate_raws(raws)
    else:
        raw = raws[0]
    return raw


def data_preprocess(raw, visualize=False):
    raw_original = raw.copy()
    filter_raw = processor.clean_eeg_data(raw, [0,16])  # 2, 4, 8, 10, 5, 9, 16

    events, e_id = processor.find_events(filter_raw)
    epochs = processor.create_epochs(filter_raw, events, e_id)
    if visualize:
        plotter(raw_original, "Raw EEG Data", show_trace=False)
        filter_plotter(raw_original, filter_raw)
        plotter(raw_original, show_trace=False, title="Original EEG")
        plotter(epochs, show_trace=False, title="Filtered EEG")

    return epochs


def train(epochs):
    X, y = trainer.get_train_data(epochs=epochs)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Class distribution: {np.bincount(y)}")

    # Train - test - valid
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42,
    )
    print("*" * 50)
    trainer.get_cv_score(X_train_val, y_train_val)
    print("*" * 50)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.15,
        stratify=y_train_val,
        random_state=43,
    )

    trainer.fit(X_train, y_train)

    val_accuracy = trainer.pipeline.score(X_val, y_val)
    print(f"Validation accuracy: {val_accuracy:.2f}")

    oconnector.save_model(trainer.pipeline, "csplda.joblib")

    # model = oconnector.load_model("csplda.joblib")
    # oconnector.get_model_info(model)

    # y_val_pred = model.predict(X_val)
    # val_acc = accuracy_score(y_val, y_val_pred)

    # y_test_pred = model.predict(X_test)
    # test_acc = accuracy_score(y_test, y_test_pred)

    # print(f"Validation accuracy: {val_acc:.4f}")
    # print(f"Test accuracy: {test_acc:.4f}")


def infer(epochs, model_path="csplda.joblib"):
    X, y = trainer.get_train_data(epochs=epochs)
    model = oconnector.load_model(model_path)

    y_pred = model.predict(X)

    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    for i, (pred, true) in enumerate(zip(y_pred, y)):
        match = pred == true
        correct += match
        print(f"epoch {i:02d}: [{pred}] [{true}] {match}")

    acc = correct / len(y_pred)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--subjects', nargs='+', type=int, required=True)
    parser.add_argument('-r', '--runs', nargs='+', type=int, required=True)
    parser.add_argument('-p', '--process', type=str, required=True, choices=['train', 'infer'])
    
    args = parser.parse_args()
    
    if len(args.subjects) < 2 and args.process == "train":
        print("Error: Minimum 2 subject IDs required.")
        sys.exit(1)
    if len(args.runs) < 4 and args.process == "train":
        print("Error: Minimum 4 runs required.")
        sys.exit(1)
        
    raw = load_data(args.subjects, args.runs)
    if args.process == "train":
        epochs = data_preprocess(raw, visualize=True)
    else:
        epochs = data_preprocess(raw, visualize=False)
    
    if args.process == "train":
        train(epochs)
    elif args.process == "infer":
        t_infer = time.perf_counter()
        infer(epochs)
        t_sum = time.perf_counter() - t_infer
        print(f"Inference time: {t_sum:.4f}s")


#  python .\mybci.py -f 3 7 -r 3 4 7 8 11 -p train,  12 infer
#  python .\mybci.py -f 3 7 -r 3 4 5 6 7 8 9 10 -p train,  11 12 infer
