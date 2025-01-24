# Configuration settings for the project.
import torch

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants for nuclear physics calculations
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]

# Random seeds for reproducibility
RANDOM_SEEDS = [1, 39, 52, 59, 71, 40]

# Path to the data files
TRAIN_DATA_PATH = "data/train_dataset.csv"
RAW_DATA_PATH = "data/raw_dataset.csv"
