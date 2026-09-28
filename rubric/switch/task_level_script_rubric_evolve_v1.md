# Task-level Script Rubric Evolver

你负责依据当前 checkpoint 最近三个训练 step 的真实 rollout，重写一份“整数据集共享”的微短片剧本 rubric。输入中的 12–16 个 case 来自互不重复的 query，每个 query 只提供一条代表性 rollout；它们是模型当前能力的横截面，不是要求你逐题定制 rubric 的模板。

只输出一个 JSON object，不要输出 Markdown fence 或 JSON 之外的文字。

## 目标

重写完整 rubric，使它在后续 step 对同一数据集的不同 query 都适用，并把评分边界移到模型当前可学习、能够区分“较差/合格/更好”的能力前沿。新 rubric 从触发 step 的下一个 step 才生效，不重打触发 step。

## 证据阅读方法

1. 同时阅读 query、rollout、旧 rubric 的逐项分数和 judge reason；不要只按总分排序。
2. 先区分：
   - `mastered`：多数 case 已稳定拿到最高档；
   - `unreachable_now`：几乎所有 case 都在最低档，正文中也没有可分档的程度差异；
   - `current_frontier`：至少若干 case 已有可复核的较高档事实，同时另一些仍停在较低档。
3. 优先收紧 mastered 的浅层存在检查，或把 unreachable 的目标改写成当前可见的必要前置能力。不得为了制造方差奖励低质量代理。
4. 新分档必须奖励正文中的功能、因果、选择、状态变化、推进或回收；关键词出现、标题格式、术语密度、场景数量、字数、镜头标签和主题自述最多只能作为低档证据，不能单独获得最高档。

## 跨 query 与媒介约束

- rubric 是 task-level：所有 criterion 必须适用于该数据集的任意合理 query。
- 不得复制 evidence 中的专名、角色、地点、道具、具体事件、对白、结局或唯一创作方案。
- 不得引入 query 未要求的固定剧情模板。
- 保留输入 `task_description` 指定的媒介边界；媒介 criterion 应奖励媒介手段的叙事功能，不能只奖励风格标签。
- 每条 criterion 只检查一个可单独改善的能力，不与其他 criterion 奖励同一升档证据。

## 不变量（违反任一项，运行时会拒绝整份候选）

1. `criteria` 数量、顺序、`id` 集合和每项 `max_points` 必须与 current rubric 完全一致。
2. 总 max_points 必须完全不变。
3. 每项必须保留非空 `check` 和 `scoring_rule`。
4. `scoring_rule` 必须覆盖从 `max_points` 到 0 的每个整数档，并为每档给出可从剧本正文引用的事实边界。
5. 不得产生语义重复 criterion；不得把两个能力塞进同一条。
6. 不得修改为 query-specific rubric。

## 输出 schema

{
  "task_id": "<与输入完全一致>",
  "rubric_version": <current version + 1>,
  "generated_at_step": <switch step>,
  "generated_from_trigger": "scheduled_recent_rollouts",
  "diagnosis": {
    "mastered": ["<跨 query 的抽象能力>"],
    "unreachable_now": ["<跨 query 的抽象能力>"],
    "current_frontier": ["<跨 query 的抽象能力边界>"],
    "rationale": "<为什么这次完整重写仍保持通用性>"
  },
  "criteria": [
    {
      "id": "<原 id>",
      "check": "<一项通用、原子、可观察的检查>",
      "scoring_rule": "<全部整数档的可观察边界>",
      "max_points": <原值>
    }
  ]
}

## 输入

### Task

- task_id: {TASK_ID}
- task_description: {TASK_DESCRIPTION}
- training_seed: {TRAINING_SEED}
- switch_step: {SWITCH_STEP}
- evidence_steps: {EVIDENCE_STEP_RANGE}

### Current rubric

{CURRENT_RUBRIC}

### Distinct-query rollout evidence

{EVIDENCE_CASES}

### Validation feedback from rejected attempt

{VALIDATION_FEEDBACK}
