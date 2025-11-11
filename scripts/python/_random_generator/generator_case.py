from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import uuid


@dataclass
class ManualDataItem:
    """A minimal manual data item container.

    - inputs: list of dict (each element is a JSON-serializable dict)
    - outputs: list of dict
    - extra: other passthrough fields stored as a dict (may be None)
    """

    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    extra: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"inputs": self.inputs, "outputs": self.outputs}
        if self.extra:
            # extra 必须是 dict，否则跳过
            if isinstance(self.extra, dict):
                d.update(self.extra)
        return d


@dataclass
class GenerateStructure:
    """Top-level generate structure.

    - manual_data: explicit list of ManualDataItem
    - extra: other top-level keys (passthrough)

    This dataclass intentionally keeps the schema minimal and explicit.
    """

    manual_data: List[ManualDataItem]
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.extra:
            if isinstance(self.extra, dict):
                out.update(self.extra)
        out["manual_data"] = [m.to_dict() for m in self.manual_data]
        return out


def json_beautify(data: Dict[str, Any]) -> str:
    """Beautify JSON according to rules:

    - Top-level: each key-value pair should occupy one line (compact representation),
      except `manual_data` which is expanded.
    - `manual_data` is a list; each item is an object. Inside each item:
      - `inputs` and `outputs` remain arrays where each element occupies one line
        (each element is printed in compact form on its own line).
      - Other fields in the item are printed compactly on a single line.

    Implementation uses placeholders and a two-pass JSON dump.
    """
    try:
        # Defensive copy
        work = json.loads(json.dumps(data, ensure_ascii=False))
        placeholders: Dict[str, str] = {}

        # 1) For each manual_data item, compact all fields except inputs/outputs.
        if "manual_data" in work and isinstance(work.get("manual_data"), list):
            for item in work["manual_data"]:
                if not isinstance(item, dict):
                    continue
                for key in list(item.keys()):
                    if key in ("inputs", "outputs"):
                        if not isinstance(item.get(key), list):
                            continue
                        new_list = []
                        for elem in item[key]:
                            ph = f"__p_{uuid.uuid4().hex}"
                            placeholders[ph] = json.dumps(elem, ensure_ascii=False, separators=(",", ":"))
                            new_list.append(ph)
                        item[key] = new_list
                    else:
                        if item.get(key) is None:
                            continue
                        ph = f"__p_{uuid.uuid4().hex}"
                        placeholders[ph] = json.dumps(item[key], ensure_ascii=False, separators=(",", ":"))
                        item[key] = ph

        # 2) For all top-level keys except manual_data, compact them as placeholders
        for k in list(work.keys()):
            if k == "manual_data":
                continue
            if work.get(k) is None:
                continue
            ph = f"__p_{uuid.uuid4().hex}"
            placeholders[ph] = json.dumps(work[k], ensure_ascii=False, separators=(",", ":"))
            work[k] = ph

        # 3) Dump with indentation so manual_data structure is expanded
        pretty = json.dumps(work, indent=2, ensure_ascii=False)

        # 4) Replace quoted placeholders with compact representations
        for ph, compact_str in placeholders.items():
            pretty = pretty.replace(f'"{ph}"', compact_str)

        return pretty
    except Exception as e:
        return f"/* json_beautify error: {e} */\n{json.dumps(data, indent=2, ensure_ascii=False)}"
