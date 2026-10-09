---
name: msmodelslim-practice-stats
description: 统计 Ascend msModelSlim 仓库 lab_practice 目录下最佳实践量化配置的"回退率"（浮点回退 FP 与升位宽回退）与性能损失（回退计算量占比、等效平均位宽），生成中文 Markdown 报告与 Excel 明细。当用户要求统计/更新/复跑 msmodelslim lab_practice 最佳实践配置的量化回退、回退率、等效位宽、性能损失分析，或 msmodelslim 仓库最佳实践更新后需要重新生成统计报告时使用。
---

# msModelSlim 最佳实践回退率统计

统计 `https://gitcode.com/Ascend/msmodelslim` 的 `lab_practice/` 下全部最佳实践 YAML：按 配置×模型 展开，逐行给出 attn/mlp/fa3/KVcache 的量化方案与回退（FP / 升位宽→目标类型），并汇总回退计算量占比与等效平均位宽。

## 执行

一条命令跑完整管线（克隆/更新仓库 → 补抓模型 config → 分析+回归校验 → 出报告）：

```bash
bash scripts/run_pipeline.sh [报告输出目录]
```

- 产物：`最佳实践回退率统计.md`、`最佳实践回退率统计_明细.xlsx`、`最佳实践回退率看板.html`（单文件交互看板，无外部依赖可离线打开，写到输出目录）。
- 工作目录由 `MSLIM_WORK` 控制（默认 `./work`），内含仓库副本、config 缓存与中间结果；重跑只增量补抓，秒级完成。
- 网络受限时：手动放置 `$MSLIM_WORK/msmodelslim`（含 lab_practice）后跳过第 1 步即可，config 缓存同理可预置到 `$MSLIM_WORK/configs/`。

## 校验

`run_analysis.py` 内置 REF 回归校验（与已交付报告的参考值逐格比对）。**必须 0 failures 才可交付**；非 0 时先排查是引擎回归还是仓库数据本身变化，后者需更新 REF 期望值。

## 维护（仓库更新后）

- 新增/修改已有架构的 yaml：直接重跑，无需改代码。
- 新架构、新 dtype、新处理器类型、fetch 失败：按 [references/methodology.md](references/methodology.md) 第四节操作。
- 判定口径、特殊架构（DeepSeek-V4 DSA、GatedDeltaNet、KDA、LatentMoE、DiT）的参数核算规则：同见 methodology.md。

## 文件说明

| 文件 | 作用 |
|---|---|
| `scripts/run_pipeline.sh` | 一键管线（4 步） |
| `scripts/fetch_configs.py` | 从 ModelScope 抓 config.json（带缓存与别名表） |
| `scripts/engine.py` | 分析引擎：yaml 解析、模块级匹配、回退判定 |
| `scripts/run_analysis.py` | 跑引擎 + REF 回归校验，产出 df.pkl |
| `scripts/gen_report.py` | 生成 md 报告与 xlsx 明细，并导出看板数据 dash_data.json |
| `scripts/gen_dashboard.py` | 生成单文件 HTML 交互看板（KPI/Top15 条形/位宽散点/分布/可排序明细表） |
| `references/methodology.md` | 统计口径与维护参考（按需查阅） |
