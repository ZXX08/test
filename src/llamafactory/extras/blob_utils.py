"""Azure Blob media reader for LlamaFactory multimodal datasets.

Environment variables:
    BLOB_CONTAINER_URL:
        Azure container URL: "https://<account>.blob.core.windows.net/<container>".
    BLOB_SAS_TOKEN:
        SAS token with or without the leading "?".
    BLOB_BASE_PREFIX:
        Optional prefix prepended to every logical media path.
    BLOB_CACHE_DIR:
        Optional cache directory for video files and other path-based readers.
        Defaults to "~/.cache/llamafactory/blob_media".
    BLOB_MAX_CONCURRENCY:
        Azure SDK concurrency for reading one blob. It applies to both image
        in-memory reads and video cache downloads. Defaults to 4.

Typical dataset paths can stay as plain paths, for example:
    /sftdata/grounding/refGeo/images/DIOR-RSVG/0001.jpg

Integration idea:
    In `llamafactory/data/mm_plugin.py`, replace direct media opens with
    `open_blob_aware_image(path)` for images and `prepare_blob_aware_video(path)`
    before calling `av.open(...)`. This keeps dataset JSON/JSONL unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO, IOBase
from pathlib import Path
from urllib.parse import quote

try:
    import warnings

    from PIL import Image
    from PIL.Image import Image as ImageObject

    # 遥感大图分辨率极高,静默 PIL 的解压炸弹警告(数据可信,非攻击)
    Image.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
except ImportError:  # pragma: no cover - LlamaFactory normally installs pillow through transformers.
    Image = None
    ImageObject = object


logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _join_blob_path(*parts: str | None) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def has_blob_config() -> bool:
    return bool(_env("BLOB_CONTAINER_URL") and _env("BLOB_SAS_TOKEN"))


@dataclass(frozen=True)
class BlobMediaConfig:
    container_url: str
    sas_token: str
    base_prefix: str = ""
    cache_dir: Path = Path("~/.cache/llamafactory/blob_media")
    max_concurrency: int = 4

    @classmethod
    def from_env(cls) -> "BlobMediaConfig":
        container_url = _env("BLOB_CONTAINER_URL").rstrip("/")
        sas_token = _env("BLOB_SAS_TOKEN").lstrip("?")
        base_prefix = _env("BLOB_BASE_PREFIX").strip("/")
        cache_dir = Path(_env("BLOB_CACHE_DIR") or "~/.cache/llamafactory/blob_media").expanduser()
        max_concurrency = int(_env("BLOB_MAX_CONCURRENCY") or "4")

        if not container_url:
            raise RuntimeError("Missing BLOB_CONTAINER_URL.")
        if not sas_token:
            raise RuntimeError("Missing BLOB_SAS_TOKEN.")

        return cls(
            container_url=container_url,
            sas_token=sas_token,
            base_prefix=base_prefix,
            cache_dir=cache_dir,
            max_concurrency=max_concurrency,
        )


def looks_like_image_path(path: str | os.PathLike[str]) -> bool:
    return Path(os.fspath(path).split("?", maxsplit=1)[0]).suffix.lower() in IMAGE_SUFFIXES


def looks_like_video_path(path: str | os.PathLike[str]) -> bool:
    return Path(os.fspath(path).split("?", maxsplit=1)[0]).suffix.lower() in VIDEO_SUFFIXES


class BlobMediaReader:
    def __init__(self, config: BlobMediaConfig | None = None) -> None:
        self.config = config or BlobMediaConfig.from_env()
        self._container_client = None

    @property
    def container_client(self):
        if self._container_client is None:
            try:
                from azure.storage.blob import ContainerClient
            except ImportError as err:
                raise RuntimeError(
                    "Azure Blob support requires `azure-storage-blob`. "
                    "Install it in the LlamaFactory environment first."
                ) from err

            self._container_client = ContainerClient.from_container_url(self.container_url_with_sas())

        return self._container_client

    def container_url_with_sas(self) -> str:
        return f"{self.config.container_url}?{self.config.sas_token}"

    def resolve_path(self, path: str | os.PathLike[str]) -> str:
        path_text = os.fspath(path).strip()
        if path_text.startswith(("http://", "https://")):
            raise ValueError("BlobMediaReader expects a logical blob path, not a full URL.")

        return _join_blob_path(self.config.base_prefix, path_text.strip("/"))

    def public_url_without_sas(self, path: str | os.PathLike[str]) -> str:
        blob_name = quote(self.resolve_path(path), safe="/")
        return f"{self.config.container_url}/{blob_name}"

    def exists(self, path: str | os.PathLike[str]) -> bool:
        return self.container_client.get_blob_client(self.resolve_path(path)).exists()

    def read_bytes(self, path: str | os.PathLike[str]) -> bytes:
        blob_name = self.resolve_path(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.download_blob(max_concurrency=self.config.max_concurrency).readall()

    def open_binary(self, path: str | os.PathLike[str]) -> BytesIO:
        stream = BytesIO(self.read_bytes(path))
        stream.name = os.fspath(path)
        return stream

    def open_image(self, path: str | os.PathLike[str]) -> "ImageObject":
        if Image is None:
            raise RuntimeError("Blob image loading requires pillow.")

        image = Image.open(self.open_binary(path))
        image.load()
        return image.convert("RGB")

    def cache_file(self, path: str | os.PathLike[str]) -> Path:
        blob_name = self.resolve_path(path)
        suffix = Path(blob_name).suffix
        cache_name = f"{hashlib.sha256(blob_name.encode('utf-8')).hexdigest()}{suffix}"
        cache_path = self.config.cache_dir / cache_name[:2] / cache_name
        if cache_path.exists():
            return cache_path

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{cache_name}.", dir=str(cache_path.parent))
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(self.read_bytes(path))
            os.replace(tmp_name, cache_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

        return cache_path


@lru_cache(maxsize=1)
def default_blob_media_reader() -> BlobMediaReader:
    return BlobMediaReader()


def should_use_blob(path: object) -> bool:
    if not isinstance(path, (str, os.PathLike)):
        return False

    path_text = os.fspath(path)
    return not path_text.startswith(("http://", "https://")) and not os.path.exists(path_text)


def open_blob_aware_image(image: str | os.PathLike[str] | bytes | IOBase | "ImageObject") -> object:
    if Image is None:
        raise RuntimeError("Image loading requires pillow.")

    if should_use_blob(image):
        logger.debug("Reading image from Azure Blob: %s", image)
        return default_blob_media_reader().open_image(image)

    if isinstance(image, bytes):
        return Image.open(BytesIO(image))
    if isinstance(image, (str, os.PathLike, IOBase)):
        return Image.open(image)

    return image


def prepare_blob_aware_video(video: str | os.PathLike[str] | IOBase) -> str | IOBase:
    if should_use_blob(video):
        logger.debug("Caching video from Azure Blob: %s", video)
        return str(default_blob_media_reader().cache_file(video))

    return video


def read_blob_media_bytes(path: str | os.PathLike[str]) -> bytes:
    return default_blob_media_reader().read_bytes(path)


def open_blob_media_stream(path: str | os.PathLike[str]) -> BytesIO:
    return default_blob_media_reader().open_binary(path)
