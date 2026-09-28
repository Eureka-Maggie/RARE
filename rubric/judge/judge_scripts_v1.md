# Judge

你要按给定 rubric 给一份候选剧本打分。

---

## 规则

- 每条 criteria 独立打分。
- Evidence 必须能在剧本原文里指到具体段落 / 具体句子 / 具体行。
- 不脑补：只根据剧本真实写了什么打分，不推测作者本意。
- 若不确定分档，从低不从高。
- Truncated 或未写完的剧本按写到的部分打分。不完整带来的逻辑不完善可在对应 rubrics 的打分中体现。

---

## 打分规程

对 rubric 的每一条 criteria：

1. 读 `check` 和 `scoring_rule`
2. 在剧本里找可以支持 / 反驳该条的 literal evidence
3. 匹配 `scoring_rule` 里的分档
4. 记录 evidence 出处（引用剧本原文的一小段）
5. 给该档对应的 int 分数

---

## 输出格式

严格 JSON，不要包裹在 markdown fence 里，第一字符是 `{`，最后是 `}`：

```
{
  "criteria_scores": [
    {"id": "c1", "score": <int>, "max": <int>, "reason": "<一句话，含剧本原文的具体引用>"},
    {"id": "c2", "score": <int>, "max": <int>, "reason": "..."},
    ...
  ],
  "total_score": <sum of score>,
  "total_max": <sum of max>
}
```

- `criteria_scores` 顺序与 rubric 中 `criteria` 顺序一致
- `reason` 必须包含从剧本引用的短句（用中文引号或英文引号包裹）作为 evidence
- `total_score` = 所有 score 之和；`total_max` = 所有 max 之和

---

## 输入区

### Query

{QUERY}

### Rubric

```json
{RUBRIC}
```

### 候选剧本

{RESPONSE}
