# RARE project page

The static project page is served at <https://eureka-maggie.github.io/RARE/>.
The [Pages workflow](../.github/workflows/pages.yml) deploys the `main` branch's
`docs/` directory when page files change, or when run manually.

Preview locally from the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://localhost:8000/>. No package installation is required.

## Authorship

Author names and order follow the [arXiv record](https://arxiv.org/abs/2610.25434).
Affiliation superscripts, contact email, and the dagger annotation follow the
supplied manuscript's title page. The dagger denotes **Project Leader**.

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

Cases are selected from a numbered text index. A compact table compares the
four evaluation dimensions; the two script columns use manuscript-style
typography with independent scrolling. On small screens the columns stack.

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

The research figures below are direct crops of the author-supplied `RARE.pdf`
(provided on 15 October 2026). The PDF's original artwork, labels, and values
are preserved. Crop rectangles use PDF points from the top-left of the page;
the PNGs are rendered at five pixels per point.

| Website asset | Paper location | Crop rectangle (x0, y0, x1, y1) |
| --- | --- | --- |
| `assets/paper-figure-1.png` | Figure 1, page 3 | 62, 84, 534, 228 |
| `assets/paper-table-2.png` | Table 2, page 6 | 62, 240, 534, 538 |
| `assets/paper-figure-5.png` | Figure 5, page 9 | 79, 401, 516, 550 |

These replace the previous `method_overview.png`, `main_results.png`, and
`pairwise_evaluation.png` assets. The root README uses the same paper crops.
Human evaluation is reported in text, and the additional-task results table
transcribes Table 4 of the paper.

The script logo is the author's supplied manuscript illustration, reused in the
page title and browser favicon. Resource buttons use the GitHub Octicons mark and
the Hugging Face logo; see `../THIRD_PARTY_NOTICES.md` for their sources.
Markdown rendering uses locally vendored Marked and DOMPurify; their license
files are in `vendor/`. The site loads no external fonts, analytics, or scripts.

The layout takes inspiration from the [MathVista](https://mathvista.github.io/)
and [We-Math](https://we-math.github.io/) project pages: compact resource buttons,
selective emphasis in the abstract, and visible experimental results.

The results include matched static-versus-RARE comparisons, pairwise model
judgments, human preference, and separately trained writing/general-generation
policies. Human preference counts ties and unresolved cases as half a win; it is
not a raw win rate. The case browser still compares RARE with the initial Base
model, not the static-rubric baseline in the aggregate results.

## Citation

The `#bibtex` section contains the official
[arXiv BibTeX entry](https://arxiv.org/bibtex/2610.25434), also stored in
[`../CITATION.bib`](../CITATION.bib). Keep this entry, the root README, and the
Hugging Face model and dataset citations in sync when publication metadata changes.
