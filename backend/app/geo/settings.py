"""Optional settings file (safeuav.toml). Everything has a default; a missing file is fine.

Only the values below can be set. They are checked here and again by `Session.configure`, and they can
never touch the trusted safety constants (no-fly zone, geofence, battery reserve, pad whitelist).
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_PORT = 8000
SESSION_KEYS = {"model": str, "mode": str, "capacity": (int, float), "wind_speed": (int, float),
                "wind_from": (int, float), "clock": str, "monitor": bool, "wait_for_ai": bool, "speed": int}


class SettingsError(Exception):
    pass


def find_file() -> Path | None:
    for p in (Path.cwd() / "safeuav.toml", Path(__file__).resolve().parents[3] / "safeuav.toml"):
        if p.is_file():
            return p
    return None


def load(path: Path | None = None) -> dict:
    """Returns {"port": int, "session": {...}}."""
    out = {"port": DEFAULT_PORT, "session": {}}
    path = path or find_file()
    if path is None:
        return out
    if sys.version_info < (3, 11):
        raise SettingsError("safeuav.toml needs Python 3.11 or newer.")
    import tomllib
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise SettingsError(f"Cannot read {path.name}: {e}") from e
    unknown = set(data) - {"server", "session"}
    if unknown:
        raise SettingsError(f"Unknown section in {path.name}: {', '.join(sorted(unknown))}")
    server = data.get("server", {})
    if set(server) - {"port"}:
        raise SettingsError("[server] only knows 'port'.")
    if "port" in server:
        port = server["port"]
        if type(port) is not int or not 1024 <= port <= 65535:
            raise SettingsError("[server] port must be a number from 1024 to 65535.")
        out["port"] = port
    for k, v in data.get("session", {}).items():
        want = SESSION_KEYS.get(k)
        if want is None:
            raise SettingsError(f"[session] has no setting '{k}'. Allowed: {', '.join(sorted(SESSION_KEYS))}.")
        if isinstance(v, bool) != (want is bool) or not isinstance(v, want):
            raise SettingsError(f"[session] {k} has the wrong type.")
        out["session"][k] = v
    return out
