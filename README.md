# msmodelslim-practice-stats

统计 [Ascend/msModelSlim](https://gitcode.com/Ascend/msmodelslim) 仓库 `lab_practice/` 目录下全部最佳实践 YAML 量化配置的**回退率**（浮点回退 FP 与升位宽回退）与**性能损失**（回退计算量占比、等效平均位宽）的 Kimi/Agent Skill。

一条命令跑完整管线，输出三件套：Markdown 统计报告 + Excel 明细 + 单文件 HTML 交互看板。

## 安装（解压即加载）

```bash
# 方式一：直接 clone 到 skills 目录（目录名即 skill 名）
git clone https://github.com/Xinhang-Li/msmodelslim-practice-stats.git ~/.user/skills/msmodelslim-practice-stats

# 方式二：下载 .skill 包解压（.skill 就是 zip）
unzip msmodelslim-practice-stats.skill -d ~/.user/skills/
```

加载判定：skills 目录下存在含 `SKILL.md`（带 name + description 头）的文件夹，agent 启动时自动注册。

## 使用

对话触发：对 agent 说「更新 msmodelslim 回退率统计」即可。

或手动执行：

```bash
bash scripts/run_pipeline.sh ./输出目录
```

环境变量：`MSLIM_WORK`（工作目录，默认 `./work`，复用则重跑约 1 分钟）、`MSLIM_REPO`、`MSLIM_OUT`。

## 产物

| 文件 | 内容 |
|---|---|
| `最佳实践回退率统计.md` | 统计口径 / 总体结论 / 性能损失分析 / 逐配置明细 / 按模型分组 |
| `最佳实践回退率统计_明细.xlsx` | 逐配置明细 / 性能损失分析 / 汇总 三个 sheet |
| `最佳实践回退率看板.html` | 单文件交互看板：KPI、Top15 条形、位宽散点、直方图、可排序/过滤/分组明细表，点击模型可看 YAML 原文，离线可开 |

## 维护

- 常规更新（新 yaml / 已有架构新配置）：直接重跑，无需改代码。
- 新架构、新 dtype、fetch 失败：按 `references/methodology.md` 第四节操作。
- 每次改动后必须确认 `run_analysis.py` 的 REF 回归校验为 0 失败。
