from .data import Data

"""Args
    path: path to the video data folder
    """
def get_data_by_path(path, train_params):
    """Return required data provider class"""
    return Data(path, **train_params)
