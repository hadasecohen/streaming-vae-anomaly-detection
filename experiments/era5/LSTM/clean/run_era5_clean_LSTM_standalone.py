
import os

from regression.run_trial import run_trial

if __name__ == "__main__":

    MODEL_ARCH = "LSTM"
    CONFIG_FILE = "era5_clean_LSTM_config_standalone.yaml"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CONFIG = os.path.join(BASE_DIR, CONFIG_FILE)
    
    run_trial(DEFAULT_CONFIG, MODEL_ARCH)
