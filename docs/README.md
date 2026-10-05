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
generated outputs. Excerpts are direct selections from those files.

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
To share one case, use `?case=live%2F016#examples`, for example.

## Assets

The method and result figures are copies of the released figures in `/assets`.
Markdown rendering uses locally vendored Marked and DOMPurify; their license
files are in `vendor/`. The site loads no external fonts, analytics, or scripts.
