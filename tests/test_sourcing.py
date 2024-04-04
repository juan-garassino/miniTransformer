from your_package.sourcing import download_data
import os


def test_download_data():
    download_data(url="http://example.com/data.csv", save_path="data.csv")
    assert os.path.exists("data.csv")
    os.remove("data.csv")  # Clean up after test
