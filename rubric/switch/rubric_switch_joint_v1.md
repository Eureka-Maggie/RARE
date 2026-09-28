# Joint rubric evolution

当前 rubric 中有多条 criterion 在同一 query 的全部 current rollouts 上得到满分。请比较全部 rollouts，在一次回复中更新其中 {UPDATE_COUNT} 条。

先找出正文中可观察的质量差别，再选择 {UPDATE_COUNT} 项最清楚、彼此独立的差别，分别对应到最合适的 saturated criterion。每项差别都需要指出表现最好和相对较弱的 rollout，并抽象成通用学习方向。

{POLARITY_INSTRUCTION}

每条 replacement 必须：

- 保留所选 saturated criterion 的 `id` 和 `max_points`；
- 只表达一个学习方向，各分档沿同一质量差别递进；
- 能依据回答正文独立评分；
- 不写入 rollout id、当前回答的专有情节或指定实现方式；
- 与其他 replacement 及 retained criteria 使用不同的升分和降分证据；
- `scoring_rule` 明确写出从满分到 0 分的每一档。

若多项差别依赖同一批升分和降分证据，只保留其中最清楚的一项。`updates` 按 Saturated criteria 中的原顺序输出。只输出 JSON。

{
  "updates": [
    {
      "id": "<selected saturated criterion id>",
      "polarity": "positive or negative",
      "diagnosis": {
        "difference": "<最好与较弱 rollouts 的一项差别>",
        "best_rollout_ids": ["R0"],
        "weaker_rollout_ids": ["R1"],
        "learning_direction": "<通用学习方向>",
        "nonredundancy": "<与其他 replacement 和 retained criteria 的评分证据区别>"
      },
      "rubric": {
        "id": "<同一 selected criterion id>",
        "check": "<一句可观察的质量检查>",
        "max_points": <逐字复制原 criterion 的整数>,
        "scoring_rule": "<max_points> = ...；逐档写到 0 = ..."
      }
    }
  ]
}

## Query

{QUERY}

## Trigger

{TRIGGER_CONTEXT}

## Current rubric

{CURRENT_RUBRIC}

## Saturated criteria

{SATURATED_CRITERIA}

## Retained criteria

{RETAINED_CRITERIA}

## Current rollouts and scores

{CURRENT_ROLLOUTS_WITH_SCORES}
