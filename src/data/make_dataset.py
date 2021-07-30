# -*- coding: utf-8 -*-
# Path: src/data/make_dataset.py

import os
from dotenv import find_dotenv, load_dotenv
import requests
import zipfile
import io

import pandas as pd

from typing import Final

# Load environment variables from .env file
load_dotenv(find_dotenv())

DATA_FILES: Final[tuple[str, ...]] = (
    str(os.getenv("APPLICATION_TEST_FILE_NAME")),
    str(os.getenv("APPLICATION_TRAIN_FILE_NAME")),
    str(os.getenv("BUREAU_BALANCE_FILE_NAME")),
    str(os.getenv("BUREAU_FILE_NAME")),
    str(os.getenv("CREDIT_CARD_BALANCE_FILE_NAME")),
    str(os.getenv("INSTALLMENTS_PAYMENTS_FILE_NAME")),
    str(os.getenv("POS_CASH_BALANCE_FILE_NAME")),
    str(os.getenv("PREVIOUS_APPLICATION_FILE_NAME")),
    str(os.getenv("SAMPLE_SUBMISSION_FILE_NAME")),
)


def download_extract_zip(zip_file_url: str, content_path: str) -> None:
    """Download Zip from url and extract content files to local path.

    - Check if content files already exist.
        - If they all exist, return.
        - If not, download zip file and extract content files.

    Args:
        zip_file_url: Url of zip file to download.
        content_path: Path to extract zip contents to.

    Returns:
        None
    """

    # We must NOT download and extract zip file by default.
    must_download: bool = False

    for file in DATA_FILES:
        # Check if content files exist
        file_path = os.path.join(content_path, file)
        if not os.path.exists(file_path):
            # If at least one file does not exist, we must download zip file
            must_download = True
            break

    # If all files already exist, return
    if not must_download:
        return

    # Download zip file
    r = requests.get(zip_file_url)
    if r.status_code != 200:
        raise ValueError(f"Could not download {zip_file_url}")

    # Check if zip file is OK
    z = zipfile.ZipFile(io.BytesIO(r.content))
    if z.testzip() is not None:
        raise ValueError(f"Could not extract {zip_file_url}")

    # Check if content path exists
    if not os.path.exists(content_path):
        os.makedirs(content_path)

    # Extract files from zip
    z.extractall(content_path)


def process_raw_files(raw_path: str, processed_path: str) -> None:
    """Read raw CSV files and save processed data as CSV.

    Args:
        raw_path: Path to raw data files.
        processed_path: Path to save processed data files.

    Returns:
        None
    """
    for file in DATA_FILES:
        # Check if processed file already exists
        processed_file_path = os.path.join(processed_path, file)
        if os.path.exists(processed_file_path):
            continue

        # Check if raw file exists
        raw_file_path = os.path.join(raw_path, file)
        if not os.path.exists(raw_file_path):
            raise ValueError(f"Could not find {raw_file_path}")

        # Read raw data
        df = pd.read_csv(raw_file_path)

        # Check if processed path exists
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)

        # Write data to CSV
        df.to_csv(processed_file_path, index=False)


def main() -> None:
    """Run data processing scripts.

    - Fix absolute files paths.
    - Download and extract data.
    - Process raw data into CSV.

    Args:
        None

    Returns:
        None
    """
    if os.getenv("ZIP_FILE_URL") is None:
        raise ValueError("ZIP_FILE_URL is not set")

    if os.getenv("RAW_DATA_PATH") is None:
        raise ValueError("RAW_DATA_PATH is not set")

    if os.getenv("PROCESSED_DATA_PATH") is None:
        raise ValueError("PROCESSED_DATA_PATH is not set")

    # Check absolute paths
    current_dir = os.path.dirname(__file__)
    src_path = os.path.abspath(os.path.join(current_dir, os.pardir))
    project_path = os.path.abspath(os.path.join(src_path, os.pardir))
    raw_data_path = os.path.join(project_path, str(os.getenv("RAW_DATA_PATH")))
    processed_data_path = os.path.join(
        project_path, str(os.getenv("PROCESSED_DATA_PATH"))
    )

    # Download zip from url and extract files to raw data path.
    download_extract_zip(
        os.path.abspath(str(os.getenv("ZIP_FILE_URL"))), raw_data_path,
    )

    # Read data into CSV
    process_raw_files(
        raw_data_path, processed_data_path,
    )


if __name__ == "__main__":
    # Run main function
    main()
