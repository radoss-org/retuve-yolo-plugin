# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from radstract.data.dicom import convert_dicom_to_images
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.keyphrases.config import Config
from retuve.logs import log_timings

from .utils import FILEDIR, shared_yolo_predict

WEIGHTS = f"{FILEDIR}/weights/v1.0/hip-yolo-us.onnx"
WEIGHTS_PATH = f"{FILEDIR}/weights/v1.0/"


# check weights file exists
if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"Weight file not found: {WEIGHTS}")


def _download_weights(api_url: str) -> bool:
    """Download weights from GitHub repository or public URL."""
    ext = api_url.split(".")[-1]
    file_path = WEIGHTS_PATH + f"hip-yolo-us.{ext}"
    os.makedirs(WEIGHTS_PATH, exist_ok=True)

    try:
        print(f"Downloading {api_url}...")
        download_url = _convert_github_url(api_url)

        if "github.com" in download_url and "/raw/" in download_url:
            headers = {"User-Agent": "Python-GitHub-File-Downloader"}
        elif "api.github.com" in download_url:
            load_dotenv()
            token = os.getenv("GITHUB_PAT")
            if not token:
                print(
                    "Warning: GITHUB_PAT not found. This may fail for private repositories..."
                )
                headers = {"User-Agent": "Python-GitHub-File-Downloader"}
            else:
                headers = {
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3.raw",
                    "User-Agent": "Python-GitHub-File-Downloader",
                }
        else:
            headers = {"User-Agent": "Python-GitHub-File-Downloader"}

        response = requests.get(download_url, headers=headers, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(file_path)
        print(f"Successfully downloaded {file_path} ({file_size:,} bytes)")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False


def _convert_github_url(url: str) -> str:
    """Convert various GitHub URL formats to downloadable URLs."""
    if "github.com" not in url:
        return url

    if "/blob/" in url:
        parts = url.replace("https://github.com/", "").split("/")
        if len(parts) >= 4:
            owner, repo = parts[0], parts[1]
            branch_and_path = "/".join(parts[3:])
            path_parts = branch_and_path.split("/")
            branch, file_path = path_parts[0], "/".join(path_parts[1:])

            load_dotenv()
            token = os.getenv("GITHUB_PAT")

            if token:
                return (
                    f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
                )
            else:
                return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    if "raw.githubusercontent.com" in url or "/raw/" in url or "api.github.com" in url:
        return url

    return url


def get_yolo_model_us(config, weights_path=None, download_if_missing=True):
    """Load YOLO model for ultrasound with automatic weight downloading."""
    from ultralytics import YOLO

    target_weights = weights_path if weights_path is not None else WEIGHTS

    # If target_weights is a URL, handle custom weight downloading
    if target_weights.startswith("http") and download_if_missing:
        print(f"Custom weights URL provided: {target_weights}")
        if not _download_weights(target_weights):
            raise FileNotFoundError(f"Failed to download weights from {target_weights}")
        # After download, use the local file path
        ext = target_weights.split(".")[-1]
        target_weights = WEIGHTS_PATH + f"hip-yolo-us.{ext}"
    elif not os.path.exists(target_weights) and download_if_missing:
        print(f"Weights file {target_weights} not found. Downloading...")
        if not _download_weights(target_weights):
            raise FileNotFoundError(f"Failed to download weights to {target_weights}")

    if not os.path.exists(target_weights):
        raise FileNotFoundError(f"Weights file {target_weights} does not exist")

    print(f"Loading YOLO model from: {target_weights}")
    model = YOLO(target_weights, task="segment")

    if "onnx" not in target_weights.lower():
        model.to(config.device)

    return model


def yolo_predict_dcm_us(
    dcm, keyphrase, model=None, custom_weights=None, imgsz=512, conf=0.6
):
    """Predict on DICOM data for ultrasound."""
    config = Config.get_config(keyphrase)
    dicom_images = convert_dicom_to_images(
        dcm,
        crop_coordinates=config.crop_coordinates,
        dicom_type=config.dicom_type,
    )
    return yolo_predict_us(dicom_images, keyphrase, model, custom_weights, imgsz, conf)


def yolo_predict_us(
    images, keyphrase, model=None, custom_weights=None, imgsz=512, conf=0.6
):
    """Predict on images for ultrasound."""
    config = Config.get_config(keyphrase)

    if model is None:
        model = get_yolo_model_us(config, custom_weights)

    weights_path = custom_weights if custom_weights is not None else WEIGHTS

    seg_results, timings = shared_yolo_predict(
        images, keyphrase, weights_path, model, config, conf=conf, imgsz=imgsz
    )

    for seg_result in seg_results:
        for seg_obj in seg_result:
            if seg_obj.empty:
                continue
            seg_obj.cls = HipLabelsUS(seg_obj.cls)

    log_timings(timings, title="Segmentation Processing:")
    return seg_results


# Check weights exist on import (optional - can be removed if too strict)
if not os.path.exists(WEIGHTS):
    print(f"Warning: {WEIGHTS} does not exist. Will attempt to download when needed.")
