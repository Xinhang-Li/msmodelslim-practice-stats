#!/bin/bash
# msModelSlim lab_practice 回退率统计一键管线
# 用法: bash run_pipeline.sh [输出目录]
# 环境变量: MSLIM_WORK(工作目录, 默认 ./work)  MSLIM_REPO(仓库目录, 默认 $MSLIM_WORK/msmodelslim)  MSLIM_OUT(报告输出目录)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MSLIM_WORK="${MSLIM_WORK:-$(pwd)/work}"
export MSLIM_REPO="${MSLIM_REPO:-$MSLIM_WORK/msmodelslim}"
export MSLIM_OUT="${1:-${MSLIM_OUT:-$(pwd)}}"
mkdir -p "$MSLIM_WORK"

echo "== [1/5] 获取/更新 msmodelslim 仓库 =="
if [ -d "$MSLIM_REPO/.git" ]; then
  git -C "$MSLIM_REPO" pull --ff-only || echo "警告: git pull 失败, 沿用本地版本"
else
  git clone --depth 1 --filter=blob:none --sparse https://gitcode.com/Ascend/msmodelslim.git "$MSLIM_REPO"
  git -C "$MSLIM_REPO" sparse-checkout set lab_practice
fi

echo "== [2/4] 抓取模型 config.json（带缓存，仅补缺） =="
python3 "$SCRIPT_DIR/fetch_configs.py"

echo "== [3/4] 运行分析引擎（含回归校验） =="
python3 "$SCRIPT_DIR/run_analysis.py"

echo "== [4/5] 生成报告（md + xlsx） =="
python3 "$SCRIPT_DIR/gen_report.py"

echo "== [5/5] 生成 HTML 看板 =="
python3 "$SCRIPT_DIR/gen_dashboard.py"

echo "完成。输出: $MSLIM_OUT/最佳实践回退率统计.md、最佳实践回退率统计_明细.xlsx、最佳实践回退率看板.html"
