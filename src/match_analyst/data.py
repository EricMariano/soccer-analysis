"""Small, revision-pinned StatsBomb Open Data client with an auditable disk cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw"
REPOSITORY_URL = "https://github.com/statsbomb/open-data"
REVISION_URL = "https://api.github.com/repos/statsbomb/open-data/commits/master"
RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


class DataError(RuntimeError):
    """An actionable data-loading failure safe to present in the interface."""


class DownloadError(DataError):
    """The source could not be downloaded or did not contain the expected data."""


class CacheError(DataError):
    """The local cache is missing, incomplete, or fails its integrity checks."""


class MatchNotFoundError(DataError):
    """The requested demonstration is absent from the real source catalogue."""


@dataclass(frozen=True)
class MatchData:
    match: dict[str, Any]
    events: list[dict[str, Any]]
    event_path: Path
    provenance: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(contents)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _load_json(contents: bytes, description: str) -> Any:
    try:
        return json.loads(contents)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DataError(f"Invalid JSON in {description}.") from exc


def read_event_tables(event_path: Path | str) -> tuple:
    """Read a cached raw event file through mplsoccer's local StatsBomb reader.

    Returns mplsoccer's four tables: events, related events, freeze frames and tactics.
    The shot validation layer intentionally also retains access to the original JSON.
    """
    from mplsoccer import Sblocal

    return Sblocal().event(str(event_path))


class StatsBombClient:
    """Fetch only requested files; cached files never require a network request.

    On first access, ``master`` is resolved to a commit SHA and written to
    ``source.json``. Every subsequent access uses that same immutable revision.
    Supplying an explicit SHA selects a separate revision directory. A corrupt
    cache is reported, never silently replaced with a different source file.
    """

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        revision: str | None = None,
        offline: bool = False,
    ) -> None:
        if revision is not None and not _SHA_PATTERN.fullmatch(revision):
            raise ValueError("revision must be a full lowercase 40-character commit SHA.")
        self.cache_dir = Path(cache_dir)
        self._revision = revision
        self.offline = offline

    @property
    def revision(self) -> str:
        return self._resolve_revision()

    def _download_bytes(self, url: str) -> bytes:
        if self.offline:
            raise CacheError("Offline mode forbids downloads; load the match online first.")
        try:
            response = requests.get(
                url,
                timeout=(10, 45),
                headers={"User-Agent": "match-analyst/0.1 (StatsBomb Open Data)"},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            raise DownloadError(
                f"Could not download {url}. Check the connection, or use a previously cached match."
            ) from exc

    def _resolve_revision(self) -> str:
        if self._revision is not None:
            return self._revision
        source_path = self.cache_dir / "source.json"
        if source_path.exists():
            try:
                source = _load_json(source_path.read_bytes(), str(source_path))
                revision = source["source_revision"]
                if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
                    raise ValueError("invalid source revision")
            except (DataError, OSError, KeyError, TypeError, ValueError) as exc:
                raise CacheError(f"Invalid revision record: {source_path}.") from exc
        else:
            if self.offline:
                raise CacheError(
                    f"No cached source revision in {self.cache_dir}. Run once online to download the demo."
                )
            payload = _load_json(self._download_bytes(REVISION_URL), REVISION_URL)
            revision = payload.get("sha") if isinstance(payload, dict) else None
            if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
                raise DownloadError("GitHub did not return a full source commit SHA.")
            try:
                _atomic_write(
                    source_path,
                    _json_bytes({
                        "repository_url": REPOSITORY_URL,
                        "source_revision": revision,
                        "resolved_at": _utc_now(),
                        "revision_url": REVISION_URL,
                    }),
                )
            except OSError as exc:
                raise CacheError(f"Cannot write the cache revision to {source_path}.") from exc
        self._revision = revision
        return revision

    def _paths(self, relative_path: str) -> tuple[Path, Path]:
        path = self.cache_dir / self.revision / relative_path
        return path, path.with_suffix(".meta.json")

    def _read_json(
        self, relative_path: str, identifiers: dict[str, int | None]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        path, metadata_path = self._paths(relative_path)
        source_url = f"{RAW_BASE_URL}/{self.revision}/data/{relative_path}"
        if path.exists() or metadata_path.exists():
            if not path.exists() or not metadata_path.exists():
                raise CacheError(
                    f"Incomplete cache at {path}. Remove this file and its .meta.json sidecar, then load online."
                )
            try:
                contents = path.read_bytes()
                metadata = _load_json(metadata_path.read_bytes(), str(metadata_path))
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be an object")
                if metadata.get("sha256") != hashlib.sha256(contents).hexdigest():
                    raise ValueError("checksum mismatch")
                if metadata.get("source_revision") != self.revision:
                    raise ValueError("revision mismatch")
                if metadata.get("source_url") != source_url:
                    raise ValueError("source URL mismatch")
                if not metadata.get("downloaded_at"):
                    raise ValueError("missing download date")
                payload = _load_json(contents, str(path))
            except (DataError, OSError, ValueError) as exc:
                raise CacheError(
                    f"Cache integrity check failed at {path}. Remove the file and its .meta.json sidecar, then load online."
                ) from exc
        else:
            if self.offline:
                raise CacheError(
                    f"File is not cached: {path}. Disable offline mode and load this match once."
                )
            contents = self._download_bytes(source_url)
            payload = _load_json(contents, source_url)
            metadata = {
                "source_name": "StatsBomb Open Data",
                "repository_url": REPOSITORY_URL,
                "license_url": f"{REPOSITORY_URL}/blob/{self.revision}/LICENSE.pdf",
                "source_url": source_url,
                "downloaded_at": _utc_now(),
                "source_revision": self.revision,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "competition_id": identifiers.get("competition_id"),
                "season_id": identifiers.get("season_id"),
                "match_id": identifiers.get("match_id"),
            }
            self._validate_records(payload, source_url)
            try:
                _atomic_write(path, contents)
                _atomic_write(metadata_path, _json_bytes(metadata))
            except OSError as exc:
                raise CacheError(f"Cannot write the data cache at {path}.") from exc
        self._validate_records(payload, str(path))
        # A caller loading events by ID may not yet know the competition/season.
        # Once load_match supplies them, preserve that additional provenance.
        enriched = {
            key: value for key, value in identifiers.items()
            if value is not None and metadata.get(key) is None
        }
        if enriched:
            metadata = {**metadata, **enriched}
            try:
                _atomic_write(metadata_path, _json_bytes(metadata))
            except OSError as exc:
                raise CacheError(f"Cannot update provenance at {metadata_path}.") from exc
        return payload, metadata

    @staticmethod
    def _validate_records(payload: Any, description: str) -> None:
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise DataError(f"Expected a JSON array of objects in {description}.")

    def competitions(self) -> list[dict[str, Any]]:
        return self._read_json("competitions.json", {})[0]

    def matches(self, competition_id: int, season_id: int) -> list[dict[str, Any]]:
        payload, _ = self._read_json(
            f"matches/{int(competition_id)}/{int(season_id)}.json",
            {"competition_id": int(competition_id), "season_id": int(season_id)},
        )
        return [
            {**match, "competition_id": int(competition_id), "season_id": int(season_id)}
            for match in payload
        ]

    def events(self, match_id: int) -> list[dict[str, Any]]:
        return self._read_json(f"events/{int(match_id)}.json", {"match_id": int(match_id)})[0]

    def file_provenance(self, relative_path: str) -> dict[str, Any]:
        """Return verified metadata for a supported source file.

        Reuses the normal cache/integrity path so manifests never record a
        sidecar whose raw file has changed. Missing files respect offline mode.
        """
        if not re.fullmatch(
            r"competitions\.json|matches/[0-9]+/[0-9]+\.json|events/[0-9]+\.json",
            relative_path,
        ):
            raise ValueError("Unsupported StatsBomb source path.")
        return dict(self._read_json(relative_path, {})[1])

    def find_demo_match(self) -> dict[str, Any]:
        competitions = [
            competition for competition in self.competitions()
            if competition.get("competition_name") == "FIFA World Cup"
            and competition.get("season_name") == "2022"
            and competition.get("competition_gender") == "male"
        ]
        for competition in competitions:
            for match in self.matches(competition["competition_id"], competition["season_id"]):
                home = match.get("home_team", {}).get("home_team_name")
                away = match.get("away_team", {}).get("away_team_name")
                if {home, away} == {"Brazil", "Serbia"}:
                    return match
        raise MatchNotFoundError(
            "Brazil v Serbia was not found in the men's FIFA World Cup 2022 catalogue."
        )

    def load_match(self, match: dict[str, Any]) -> MatchData:
        try:
            match_id = int(match["match_id"])
            competition_id = int(
                match.get("competition_id") or match["competition"]["competition_id"]
            )
            season_id = int(match.get("season_id") or match["season"]["season_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError("A match must identify its match, competition and season.") from exc
        events, provenance = self._read_json(
            f"events/{match_id}.json",
            {"competition_id": competition_id, "season_id": season_id, "match_id": match_id},
        )
        event_path, _ = self._paths(f"events/{match_id}.json")
        return MatchData(match=match, events=events, event_path=event_path, provenance=provenance)
