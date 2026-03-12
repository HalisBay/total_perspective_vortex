import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_service.data_layer.data_connector import load_subdataset
from ml_service.applications.visualize import plotter


def main_function():
    raws = load_subdataset()
    plotter(raws)


if __name__ == "__main__":
    main_function()