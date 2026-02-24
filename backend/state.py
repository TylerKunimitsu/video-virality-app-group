import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

#data_path = ROOT_DIR/"USvideos.csv"
data_path = ROOT_DIR/"someUSvideos.csv"
data = pd.DataFrame(pd.read_csv(data_path))