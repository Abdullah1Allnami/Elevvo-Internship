import os
from kaggle.api.kaggle_api_extended import KaggleApi


def data_downloader():
    """
    Downloads the IMDb Dataset of 50K Movie Reviews from Kaggle
    and unzips it into the 'data' directory.
    Make sure you've authenticated using Kaggle API before running this.
    """
    os.makedirs("data", exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews", path="data", unzip=True
    )
    print('IMDb Movie Reviews dataset downloaded and unzipped to "data" directory')
