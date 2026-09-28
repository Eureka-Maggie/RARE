# Score multiple replacement rubrics on one rollout group

使用输入中的全部 rubrics 分别评价 R0–R7。每条 rubric、每条 rollout 均按照明文条件独立判断，允许相同分数，不要求制造分差。

只依据 rollout 正文中的可观察内容。`reason` 用一句话说明当前分档依据。criteria 顺序必须与输入 Rubrics 一致，每条内部按 R0–R7 顺序输出。只输出 JSON。

{
  "criteria": [
    {
      "id": "<rubric id>",
      "scores": [
        {"rollout_id": "R0", "score": 0, "reason": "<简短依据>"},
        {"rollout_id": "R1", "score": 0, "reason": "<简短依据>"},
        {"rollout_id": "R2", "score": 0, "reason": "<简短依据>"},
        {"rollout_id": "R3", "score": 0, "reason": "<简短依据>"},
        {"rollout_id": "R4", "score": 0, "reason": "<简短依据>"},
        {"rollout_id": "R5", "score": 0, "reason": "<简短依据>"},
        {"rollout_id": "R6", "score": 0, "reason": "<简短依据>"},
        {"rollout_id": "R7", "score": 0, "reason": "<简短依据>"}
      ]
    }
  ]
}

## Query

{QUERY}

## Rubrics

{RUBRICS}

## Rollouts

{ROLLOUTS}
