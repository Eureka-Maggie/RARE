# RARE data

The released query-rubric annotations are hosted at
[`EurekaTian/RARE`](https://huggingface.co/datasets/EurekaTian/RARE) on
Hugging Face. Download them into the layout expected by the training launchers:

```bash
python scripts/download_data.py
```

| Production style | Train | Test | Source-video groups |
| --- | ---: | ---: | ---: |
| 2D animation | 2,125 | 237 | 370 |
| 3D/stop-motion | 1,759 | 192 | 302 |
| Live action | 1,509 | 167 | 258 |
| **Total** | **5,393** | **596** | **930** |

## Provenance

The source videos come from the media collection used by
[AVA-Encoder](https://arxiv.org/abs/2608.12313) and cover the three production
styles above. Complete videos were processed to reconstruct detailed scripts;
five to seven creative queries were then derived from each script. Each
query-specific rubric was constructed from the query and eight pregenerated
candidate scripts so that its criteria capture observable quality differences.

The splits are group-disjoint: all queries derived from one source video stay
on the same side of the train-test boundary. The public files retain the exact
training schema while replacing source-video titles and uploader handles with
stable anonymous identifiers.

The release contains annotations only. Original videos are not redistributed.
The annotation files are licensed under CC BY 4.0; the Apache-2.0 license in
the repository root applies to code, not to third-party source material.
