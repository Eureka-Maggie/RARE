"""Build the project's nine-case browser data from checked-in original scripts."""

import json
import re
from collections import Counter
from pathlib import Path


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
            path = f"cases/{variant}/{case['id'].replace('/', '-')}.md"
            markdown = (docs / path).read_text(encoding="utf-8")
            title = re.sub(r"^#\s*微型剧本[：:]\s*", "", markdown.splitlines()[0]).strip()
            scenes = {
                int(match.group(1)): match.group(0).strip()
                for match in re.finditer(
                    r"^### S(\d+).*?(?=^### S\d+|\Z)", markdown, re.M | re.S
                )
            }
            selected = case["excerptScenes"][variant]
            case[variant] = {
                "title": title,
                "markdown": markdown,
                "excerpt": "\n\n".join(scenes[index] for index in selected),
                "sceneLabel": " + ".join(f"S{index:02d}" for index in selected),
                "download": path,
            }
    output = docs / "assets/cases.json"
    output.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"Built {len(cases)} selected pairs → {output.relative_to(docs.parent)}")


if __name__ == "__main__":
    main()
