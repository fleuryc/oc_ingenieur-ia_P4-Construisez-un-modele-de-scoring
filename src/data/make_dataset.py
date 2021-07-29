# -*- coding: utf-8 -*-
# Path: src/data/make_dataset.py

import os
from dotenv import find_dotenv, load_dotenv
import requests
import zipfile
import io


def download_extract_zip(zip_file_url: str, raw_data_path: str) -> None:
    """Downloads a zip file from a given url
    and saves its content files to local path.
    """
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(raw_data_path)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    download_extract_zip(
        os.getenv('ZIP_FILE_URL'),
        os.getenv('RAW_DATA_PATH')
    )


if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    # Run main function
    main()
