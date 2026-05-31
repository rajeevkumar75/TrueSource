from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HUMAN_ALIASES = frozenset({"human", "real", "genuine", "organic", "0", "false", "no"})
AI_ALIASES = frozenset({"ai", "fake", "machine", "generated", "synthetic", "gpt", "1", "true", "yes"})


@dataclass(frozen=True)
class TextRecord:
    text: str
    label: str  # "human" or "ai"
    source: str


def normalize_label(raw: object) -> str:
    if raw is None:
        raise ValueError("Missing label")
    value = str(raw).strip().lower()
    if value in HUMAN_ALIASES:
        return "human"
    if value in AI_ALIASES:
        return "ai"
    raise ValueError(f"Unrecognized label: {raw!r}. Use human/ai (or aliases).")


def _record_from_obj(obj: dict, source: str) -> TextRecord:
    text = obj.get("text") or obj.get("content") or obj.get("body")
    if not text or not str(text).strip():
        raise ValueError(f"Missing text in {source}")
    label_raw = obj.get("label")
    if label_raw is None:
        label_raw = obj.get("source_type") or obj.get("class")
    return TextRecord(text=str(text).strip(), label=normalize_label(label_raw), source=source)


def load_jsonl(path: Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Each JSONL line must be an object (line {line_no} in {path})")
            records.append(_record_from_obj(obj, source=f"{path.name}:{line_no}"))
    return records


def load_from_class_folders(root: Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    folder_map = {
        "human": "human",
        "real": "human",
        "ai": "ai",
        "fake": "ai",
        "generated": "ai",
    }
    for folder_name, label in folder_map.items():
        folder = root / folder_name
        if not folder.is_dir():
            continue
        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".txt", ".md", ".json"}:
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            records.append(
                TextRecord(text=text, label=label, source=str(file_path.relative_to(root)))
            )
    return records


def load_text_dataset(data_path: str | Path) -> list[TextRecord]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path.resolve()}")

    records: list[TextRecord] = []
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got: {path}")
        records.extend(load_jsonl(path))
    elif path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            for jsonl_file in jsonl_files:
                records.extend(load_jsonl(jsonl_file))
        records.extend(load_from_class_folders(path))

    if not records:
        raise ValueError(
            f"No training samples found under {path.resolve()}. "
            "Add JSONL lines like "
            '{"text": "your passage...", "label": "human"} or '
            '{"text": "...", "label": "ai"}, '
            "or create human/ and ai/ folders with .txt files."
        )
    return records
