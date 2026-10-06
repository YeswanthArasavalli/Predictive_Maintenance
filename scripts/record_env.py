"""Record the execution environment for reproducibility (§19).

Run:  python scripts/record_env.py
Writes results/audits/environment.json
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "audits" / "environment.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

PKG_NAMES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "pyarrow", "joblib", "pytest"]


def _versions() -> dict:
    out = {}
    for name in PKG_NAMES:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def main() -> int:
    env = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
        "packages": _versions(),
        "random_seed_policy": {
            "global_seed": 42,
            "note": "PYTHONHASHSEED deterministic; numpy default_rng(42) for sampling; "
            "explicit random_state for any learner/splitter in later phases.",
        },
    }
    OUT.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
