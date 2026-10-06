"""Dataset integrity verification via SHA-256 checksums.

The immutable raw files are described by ``data/raw/.checksums.txt`` (the single
source of truth, produced at copy time). This module recomputes SHA-256 over the
current on-disk files and reports, per file, whether it matches the recorded
value. It never mutates ``data/raw``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKSUM_FILE = ROOT / "data" / ".checksums.txt"
from .loader import raw_dir  # noqa: E402  (import after ROOT to avoid cycle noise)


@dataclass
class FileIntegrity:
    name: str
    recorded: str
    actual: str | None
    exists: bool
    match: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_recorded_checksums(checksum_file: Path = CHECKSUM_FILE) -> dict[str, str]:
    """Parse ``<hash> *<name>`` lines (sha256sum -c format)."""
    recorded: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "*" not in line:
            continue
        digest, name = line.split("*", 1)
        recorded[name.strip()] = digest.strip().lower()
    return recorded


def verify_raw(raw_path: Path | None = None) -> list[FileIntegrity]:
    """Recompute and compare every recorded checksum against the on-disk file."""
    raw_path = raw_path or raw_dir()
    recorded = read_recorded_checksums()
    results: list[FileIntegrity] = []
    for name, digest in recorded.items():
        fp = raw_path / name
        exists = fp.exists()
        actual = _sha256(fp) if exists else None
        results.append(
            FileIntegrity(
                name=name,
                recorded=digest,
                actual=actual,
                exists=exists,
                match=exists and actual == digest,
            )
        )
    return results


def all_match(raw_path: Path | None = None) -> bool:
    return all(r.match for r in verify_raw(raw_path))
