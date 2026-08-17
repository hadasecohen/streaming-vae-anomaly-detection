
import os

from regression.run_trial import run_trial

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CONFIG = os.path.join(BASE_DIR, "era5_group_LSTM_config_standalone.yaml")
    MODEL_ARCH = "LSTM"

    run_trial(DEFAULT_CONFIG, MODEL_ARCH)
