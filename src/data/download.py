from pathlib import Path
import kagglehub


def download_raw_dataset(dataset_name: str = "ayushmandatta1/deepdetect-2025") -> Path:
    """
    Downloads the raw dataset from Kaggle using kagglehub and returns the path to the source data directory.
    """
    path = kagglehub.dataset_download(dataset_name)
    ddata = Path(path) / "ddata"

    print(f"Download completed at: {path}")
    print(f"Source folder (ddata): {ddata}")

    return ddata
  
if __name__ == "__main__":
    download_raw_dataset()
