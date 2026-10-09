# 统计口径与维护参考

本文件供执行/维护 msmodelslim-practice-stats 时按需查阅。核心代码在 `scripts/engine.py`（分析引擎）与 `scripts/gen_report.py`（报告生成），本文档解释其中的判定规则与特殊架构处理，以及仓库更新后的维护动作。

## 一、回退判定口径

- **位宽表** `BITS`：int4/mxfp4/fp4=4，int8/mxfp8/fp8_e4m3/fp8=8，bfloat16/float16=16。新增 dtype 时必须加入此表。
- **label 主位宽**：取自 yaml `metadata.label`（w_bit/a_bit），kv_cache 标记决定 c8，fa3 由 `fa3_quant` 处理器判定。
- **浮点回退（FP）**：模块未被任何量化块（`linear_quant`/`autoround_quant`/`trainable_linear_quant`）覆盖，保持浮点。未覆盖模块在**权重侧与激活侧同时计回退**（仅 label a16 的配置激活 FP16 属计划目标，不计）。
- **升位宽回退**：模块被量化但 dtype 位宽 > label 主位宽（权重对 w_bit、激活对 a_bit），需注明回退目标类型。低于 label 位宽的属超额压缩，不是回退，只在备注提示。
- **匹配语义**：GroupProcessor first-match-wins；`_install_quantizer` 只匹配 nn.Linear；死块（未新认领任何模块的量化块）需在备注标注"量化块实际未生效"。
- **V0 旧格式**（modelslim_v0）：按 `calib_cfg.disable_names` 名单判定回退。
- **多数派规则**：某类模块回退合计 ≥99.9% 且某一项严格过半时，多数派显示为"量化方案"、少数派列为回退；50/50 不触发。
- **100% 回退简化**：全 FP → 方案写"不量化（FP16）"且不计入第二节回退统计；全部升到同一 dtype → 方案直接写该 dtype。
- **目标即 w16a16 的配置**（如纯浮点稀疏）：不做位宽量化，不属于回退统计口径——FP 不计为回退（fp_share 置 NaN，性能表/看板显示 `—`），不进入 Top15 排序与 FP 均值统计；在 gen_report.py 的 df_rank 阶段处理。

## 二、模块分类与合并约定

- **attn linear**：q/k/v/o（MLA 含 q_a/q_b/kv_a/kv_b，V4 含 wq_a/wq_b/wkv/wo_a/wo_b/compressor/indexer）**+ 线性注意力投影**（in_proj/out_proj 等，以"FP:线性N层"片段并入 attn 列，不计入 attn 多数派判定的分母）。
- **mlp linear**：dense MLP **+ 路由专家 + 共享专家**（Kimi-K3 LatentMoE 的 routed_expert_down/up_proj 名字含 'experts'，随专家一起被量化）。
- **fa3 / KVcache** 均为激活侧对象，无权重对应行。

## 三、特殊架构参数核算（engine.layer_modules / mspec_from_config）

| 架构 | 要点 |
|---|---|
| DeepSeek-V4 | 按官方 inference/model.py：wq_a(h→ql)、wq_b(ql→nh·512)、wkv(h→512)、wo_a/wo_b(分组低秩 O)；compressor 按 compress_ratios 逐层（ratio4  coff=2，ratio128 coff=1），indexer 仅存在于 ratio4 层 |
| Qwen3.5/3.6 线性注意力 | GatedDeltaNet：in_proj_qkv/z/b/a + out_proj，key_dim/value_dim 由 linear_key/value_head_dim×num_heads 计算 |
| Kimi KDA | q/k/v_proj + f_a/f_b 低秩 + b_proj + g_proj（或 g_a/g_b）+ o_proj |
| GLM-5.3 | Glm5NextTextLinearAttention，forget_gate 低秩 + g_a/g_b |
| Kimi-K3 MoE | LatentMoE：专家 hidden=routed_expert_hidden_size（3584）而非模型 hidden（7168），漏掉会 2× 高估 |
| DiT（Wan/FLUX/HunyuanVideo/Qwen-Image） | 走 DIT_CFG 静态表；verified_model_types 别名（Wan2_1↔Wan2.1）已在 get_model_types 归一；ffn 拆 ffn.0/ffn.2 两个模块；mod 调制层仅备注 |
| 多模态 | vision 塔不量化，备注"vision未量化" |

## 四、仓库更新后的维护动作

1. **常规更新**（新 yaml、已有架构的新配置）：直接重跑 `run_pipeline.sh`，fetch_configs 自动补抓新模型的 config.json，无需改代码。
2. **新模型 fetch 失败**：在 `fetch_configs.py` 的 `ALIAS` 表补 ModelScope 路径映射。
3. **新架构**（引擎输出参数 N/A 或行数异常）：在 `engine.py` 的 `mspec_from_config` / `layer_modules` 增加该架构的模块清单，参数维度以官方 modeling 代码为准，禁止拍脑袋。
4. **新处理器/dtype**：扩 `LIN_TYPES` / `BITS`。
5. **每次改动后必须看 run_analysis.py 的 REF 回归校验输出**（0 failures 才可交付）；架构改动需同步更新 REF 期望值。

## 五、输出物

- `最佳实践回退率统计.md`：四节（统计口径 / 总体结论 / 性能损失：计算量占比+等效平均位宽 / 逐配置明细 128 行级）。
- `最佳实践回退率统计_明细.xlsx`：逐配置明细 / 性能损失分析 / 汇总 三个 sheet。
- `最佳实践回退率看板.html`：单文件交互看板（KPI 指标条、回退占比 Top15 堆叠条形、目标位宽 vs 等效位宽散点、等效位宽分布、可排序/过滤明细表，支持「按模型分组」视图——分组键 fam()：模型名去尺寸/微调/日期后缀），无外部依赖可离线打开；数据源为 gen_report.py 导出的 `dash_data.json`（与 md 表四同一批单元格文本，保证口径一致）。md 第五节为同一分组键的聚合表。
- 性能口径：计算量占比 = 回退模块参数 ÷ 全模型线性层参数（≈每 token GEMM 占比）；等效位宽 = 参数加权平均（FP 按 16bit）。
