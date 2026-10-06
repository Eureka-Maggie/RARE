"""Build bilingual showcase data from original scripts and English translations."""

import json
import re
from collections import Counter
from pathlib import Path


def read_script(docs, case, variant, language):
    prefix = "cases/en" if language == "en" else "cases"
    path = f"{prefix}/{variant}/{case['id'].replace('/', '-')}.md"
    markdown = (docs / path).read_text(encoding="utf-8")
    title = re.sub(r"^#\s*(?:微型剧本[：:]|Short screenplay:)\s*", "", markdown.splitlines()[0]).strip()
    scenes = {
        int(match.group(1)): match.group(0).strip()
        for match in re.finditer(r"^### S(\d+).*?(?=^### S\d+|\Z)", markdown, re.M | re.S)
    }
    selected = case["excerptScenes"][variant]
    return {
        "title": title,
        "markdown": markdown,
        "excerpt": "\n\n".join(scenes[index] for index in selected),
        "sceneLabel": " + ".join(f"S{index:02d}" for index in selected),
        "download": path,
    }


def scene_timestamps(markdown):
    return re.findall(r"^### S(\d+)[^\n]*\[([^\]]+)\]", markdown, re.M)


def main():
    docs = Path(__file__).resolve().parents[1] / "docs"
    data = json.loads((docs / "cases/selection.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    if Counter(case["domain"] for case in cases) != {"2d": 3, "3d": 3, "live": 3}:
        raise ValueError("The showcase requires three selected cases per task.")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique.")
    for case in cases:
        for variant in ("base", "rare"):
            original = read_script(docs, case, variant, "zh")
            english = read_script(docs, case, variant, "en")
            if scene_timestamps(original["markdown"]) != scene_timestamps(english["markdown"]):
                raise ValueError(f"Scene numbers or timestamps differ: {case['id']} / {variant}")
            original["en"] = english
            case[variant] = original
    data["translation_note"] = "English translations of the Chinese outputs; scores evaluate the originals."
    output = docs / "assets/cases.json"
    output.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"Built {len(cases)} bilingual pairs → {output.relative_to(docs.parent)}")


if __name__ == "__main__":
    main()
