from src.data_preparation import load_and_prepare_data

data_path = "/kaggle/input/home-credit-credit-risk-model-stability/"

X, y = load_and_prepare_data(data_path)
