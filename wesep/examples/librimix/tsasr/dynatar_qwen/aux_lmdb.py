from __future__ import annotations

import io
import math
import pickle
import shutil
from pathlib import Path
from typing import Mapping

import lmdb
import numpy as np


DEFAULT_LMDB_MAP_SIZE = int(math.pow(1024, 4))  # 1 TB
DEFAULT_COMMIT_INTERVAL = 1000


def resolve_sidecar_path(base_path: Path, value: str) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_path.parent / candidate
    return str(candidate.resolve(strict=False))


def encode_aux_payload(payload: Mapping[str, object]) -> bytes:
    normalized_payload = {}
    for name, value in payload.items():
        normalized_payload[name] = value if isinstance(value, np.ndarray) else np.asarray(value)
    buffer = io.BytesIO()
    np.savez(buffer, **normalized_payload)
    return buffer.getvalue()


def decode_aux_payload_bytes(payload_bytes: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def open_readonly_lmdb(lmdb_path: str | Path) -> lmdb.Environment:
    path = Path(lmdb_path)
    subdir = path.is_dir() if path.exists() else True
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=subdir,
        max_readers=512,
    )


def read_lmdb_keys(lmdb_path: str | Path) -> list[str]:
    env = open_readonly_lmdb(lmdb_path)
    try:
        with env.begin(write=False) as txn:
            payload = txn.get(b"__keys__")
            if payload is not None:
                return list(pickle.loads(payload))
            return [key.decode("utf-8") for key, _ in txn.cursor() if key != b"__keys__"]
    finally:
        env.close()


def remove_lmdb_path(lmdb_path: str | Path) -> None:
    path = Path(lmdb_path)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


class LMDBWriter:
    def __init__(
        self,
        lmdb_path: str | Path,
        *,
        map_size: int = DEFAULT_LMDB_MAP_SIZE,
        commit_interval: int = DEFAULT_COMMIT_INTERVAL,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(lmdb_path)
        if overwrite:
            remove_lmdb_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            map_size=int(map_size),
            subdir=True,
            lock=True,
            readonly=False,
            create=True,
        )
        self.commit_interval = max(int(commit_interval), 1)
        self.txn = self.env.begin(write=True)
        self.keys: list[str] = []
        self.count = 0

    def put(self, key: str, payload_bytes: bytes) -> None:
        self.txn.put(key.encode("utf-8"), payload_bytes)
        self.keys.append(key)
        self.count += 1
        if self.count % self.commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        if self.txn is not None:
            self.txn.commit()
            self.txn = None
        with self.env.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(self.keys))
        self.env.sync()
        self.env.close()

    def __enter__(self) -> "LMDBWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        if self.txn is not None:
            self.txn.abort()
            self.txn = None
        self.env.close()
