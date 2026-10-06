# RARE project page

The static project page is served at <https://eureka-maggie.github.io/RARE/>
from the `main` branch's `/docs` directory using GitHub Pages.

Preview locally from the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://localhost:8000/>. No package installation is required.

## Selected examples

`cases/selection.json` records the nine author-selected examples (three per
task), original queries, paired case-export scores, English prompt
translations, descriptive browsing labels, and excerpt scene numbers.
The Markdown files in `cases/base/` and `cases/rare/` preserve the complete
generated Chinese outputs. Complete English translations are in
`cases/en/base/` and `cases/en/rare/`, including cast, scene lists, and
audiovisual notes. Excerpts are direct selections from the corresponding files.

The examples open in English. The EN / 中文 control switches all prompts,
case labels, excerpts, full scripts, and downloads together. The choice persists
across tasks and visits; an explicit `lang` URL parameter takes precedence.
English text is a translation of the original output, not a new model sample.
Scores always refer to the Chinese originals.

- **Base:** initial Qwen3-4B-Instruct-2507, before task-specific training.
- **RARE:** the sample-level Negative replacement variant.
- **Case scores:** individual output scores from the paired export.
- **Results section:** aggregate manuscript results; RL scores average four
  training seeds, and the displayed overall mean averages the three tasks.

Rebuild browser data after editing the selection metadata or source scripts:

```bash
python3 scripts/build_showcase.py
```

The generated `assets/cases.json` is checked in so Pages needs no build tools.
The build checks that scene numbers and timecodes match across languages.
To share one case, use `?case=live%2F016&lang=en#examples`, for example
(or `lang=zh` for the Chinese original).

## Assets

The method and result figures are copies of the released figures in `/assets`.
Markdown rendering uses locally vendored Marked and DOMPurify; their license
files are in `vendor/`. The site loads no external fonts, analytics, or scripts.
