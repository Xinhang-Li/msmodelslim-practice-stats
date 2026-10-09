# -*- coding: utf-8 -*-
import sys, json
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get('MSLIM_WORK', os.path.join(os.getcwd(), 'work'))
REPO = os.environ.get('MSLIM_REPO', os.path.join(WORK, 'msmodelslim'))
import engine
import pandas as pd

data = engine.load_yamls(os.path.join(REPO, 'lab_practice'))
found = json.load(open(os.path.join(WORK, 'found_index.json')))
rows=[]
for p in sorted(data):
    rows += engine.analyze(p, data, found)
df = pd.DataFrame(rows)
df.to_pickle(os.path.join(WORK, 'df.pkl'))
print('rows:', len(df), 'yamls:', df.yaml.nunique())

# ---------------- 校验：与已交付报告的参考值比对 ----------------
REF = [
 # (yaml片段, 列, 期望值子串)
 ('deepseek_v3_2/deepseek_w4a8.yaml','attn_fp_layers',62),
 ('deepseek_v3_2/deepseek_w4a8.yaml','attn_fp_params',5704777728),
 ('deepseek_v3_2/deepseek_w4a8.yaml','attn_up','62层/9.62B→int8'),
 ('deepseek_v3_2/deepseek_w4a8.yaml','moe_up','1层/11.27B→int8'),
 ('deepseek_v3_2/deepseek_w4a8.yaml','mlp_up','3层/1.19B→int8'),
 ('deepseek_v3/deepseek_w4a8_per_channel.yaml','attn_fp_params',1040187392),
 ('deepseek_v3/deepseek_w4a8_per_channel.yaml','attn_up','62层/10.56B→int8'),
 ('deepseek_v3/deepseekv31_terminus_w4a4c8.yaml','attn_up','62层/10.56B→mxfp8'),
 ('kimi_k2_5/kimi_k2_6_w4a8.yaml','attn_fp_layers',61),
 ('kimi_k2_5/kimi_k2_6_w4a8.yaml','attn_fp_params',511705088),
 ('kimi_k2_5/kimi_k2_6_w4a8.yaml','attn_up','61层/5.66B→int8'),
 ('kimi_k2_5/kimi_k2_6_w4a8.yaml','mlp_up','1层/0.40B→int8'),
 ('gemma4/gemma4_mxfp8_mxfp4.yaml','attn_up','60层/7.71B→mxfp8'),
 ('gemma4/gemma4_mxfp8_mxfp4.yaml','mlp_fp_layers',6),
 ('wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml','attn_up','5层/1.05B→mxfp8'),
 ('wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml','mlp_up','5层/0.71B→mxfp8'),
 ('wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml','fa3_up','1层→fp8_e4m3'),
 ('wan2_2/wan2_2_w4a4f8_mxfp_t2v.yaml','fa3_fp_layers',1),
 ('hunyuan_video/hunyuan_video_w8a8f8_mxfp.yaml','note','调制层(mod)浮点回退60块'),
 ('qwen_image_edit/qwen-image-edit-w4a4f4-mxfp.yaml','note','调制层(mod)浮点回退60块'),
 ('qwen2/qwen2-72b-w8a8c8.yaml','mlp_fp_layers',80),
 ('qwen2/qwen2-72b-w8a8c8.yaml','mlp_fp_params',19377684480),
 ('qwen2/qwen2-72b-w8a8c8.yaml','kvc_rb_layers',0),
 ('qwen3_moe/qwen3-coder-480b-w4a8.yaml','attn_up','62层/10.14B→int8'),
 ('qwen3_moe/qwen3-coder-480b-w4a8.yaml','moe_fp_layers',10),
 ('qwen3_moe/qwen3-coder-480b-w4a8.yaml','moe_fp_params',75497472000),
 ('glm4_moe/glm4_7_moe-w8a8-v1.yaml','attn_fp_layers',1),
 ('glm4_moe/glm4_7_moe-w8a8-v1.yaml','moe_fp_layers',1),
 ('glm4_moe/glm4_7_moe-w8a8-v1.yaml','mlp_fp_layers',3),
 ('glm4_moe/glm4_7_moe-w8a8-v1.yaml','shared_fp_layers',90),
 ('glm_5/glm_5_1_w4a8.yaml','attn_fp_layers',79),
 ('glm_5/glm_5_1_w4a8.yaml','attn_up','79层/13.86B→int8'),
 ('glm_5/glm_5_1_w4a8.yaml','mlp_up','3层/0.68B→int8'),
 ('glm_5_2/glm_5_2_w8a8c8.yaml','attn_fp_layers',79),
 ('glm_5_2/glm_5_2_w8a8c8.yaml','fa3_fp_layers',8),
 ('glm_5_2/glm_5_2_w8a8c8.yaml','kvc_rb_layers',9),
 ('deepseek_v4/deepseek_v4_pro_w4a8.yaml','attn_fp_layers',62),
 ('deepseek_v4/deepseek_v4_flash_w8a8.yaml','attn_fp_layers',44),
 ('qwq/qwq-32b-w8a8.yaml','mlp_fp_layers',64),
 ('deepseek-r1-distill/deepseek-r1-distill-qwen-1.5b-w8a8.yaml','mlp_fp_layers',28),
]
fails=[]
for y,col,exp in REF:
    sub = df[df.yaml==y]
    if len(sub)==0: fails.append((y,col,'行缺失')); continue
    v = sub.iloc[0][col]
    ok = (str(exp) in str(v)) if isinstance(exp,str) else (v==exp)
    if not ok: fails.append((y,col,f'期望{exp} 实际{v}'))
print('校验失败项:', len(fails))
for f in fails: print(' ', f)

# 汇总计数（仅展示，不做断言——随仓库更新会合理漂移；回归防护以上面的单元格级 REF 断言为准）
def n_fp(c): return int((df[c].fillna(0)>0).sum())
def has_up(c): return int(df[c].fillna('').astype(str).str.strip().ne('').sum())
print('attn_fp',n_fp('attn_fp_layers'),' attn_up',has_up('attn_up'))
print('moe_fp',n_fp('moe_fp_layers'),' moe_up',has_up('moe_up'))
print('mlp_fp',n_fp('mlp_fp_layers'),' mlp_up',has_up('mlp_up'))
print('fa3_en',int(df.fa3_fp_layers.notna().sum()),' fa3_fp',n_fp('fa3_fp_layers'),' fa3_up',has_up('fa3_up'))
print('kvc_en',int(df.kvc_rb_layers.notna().sum()),' kvc_fp',n_fp('kvc_rb_layers'))
