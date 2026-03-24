import joblib
from pathlib import Path


class ObjectConnector:
    def __init__(self, model_dir="model"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    def save_model(self, model, file_name):
        file_path = self.model_dir / file_name
        joblib.dump(model, file_path)
        print(f"model saved to {file_path}")

    def load_model(self, file_name):
        file_path = self.model_dir / file_name
        print("model loaded")
        return joblib.load(file_path)

    def get_model_info(self, model):
        print("*" * 50, "\n")
        print(f"model params : {model.get_params()}\n\n")
        print("*" * 50, "\n")