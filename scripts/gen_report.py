# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get('MSLIM_WORK', os.path.join(os.getcwd(), 'work'))
OUT = os.environ.get('MSLIM_OUT', os.getcwd())
import pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BITS_D = {'int4':4,'mxfp4':4,'fp4':4,'int8':8,'mxfp8':8,'fp8_e4m3':8,'fp8':8,'bfloat16':16,'float16':16}

df = pd.read_pickle(os.path.join(WORK, 'df.pkl')).sort_values('yaml').reset_index(drop=True)
for c in ['attn_up','moe_up','mlp_up']:
    df[c]=df[c].astype(str).str.replace('/0.00B→','→',regex=False).replace('nan','')

def par2b(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return None
    if isinstance(v,(int,float)): return round(v/1e9,2)
    s=str(v)
    if s=='N/A': return 'N/A'
    if s.endswith('+'):
        try: return f"{float(s[:-1])/1e9:.2f}+"
        except: return s
    return s

def fmt_cell(fp_l, fp_p, up):
    parts=[]
    na = fp_l is None or (isinstance(fp_l,float) and np.isnan(fp_l))
    if not na and fp_l:
        p=par2b(fp_p)
        parts.append(f"FP:{int(fp_l)}层" + (f"/{p}B" if isinstance(p,(int,float)) else (f"/{p}" if isinstance(p,str) and p else "")))
    if isinstance(up,str) and up.strip():
        for seg in up.split('; '): parts.append('升位宽:'+seg)
    if parts: return '；'.join(parts)
    return 'N/A' if na and str(fp_p)=='N/A' else '—'


def pct1(x):
    s = f"{x:.1f}"
    return '<0.1' if s == '0.0' and x > 0 else s

def fa3_cell(r):
    if pd.isna(r['fa3_fp_layers']): return '未启用'
    parts=[]
    if r['fa3_fp_layers']: parts.append(f"FP:{int(r['fa3_fp_layers'])}层")
    if isinstance(r['fa3_up'],str) and r['fa3_up'].strip(): parts.append('升位宽:'+r['fa3_up'])
    return '；'.join(parts) if parts else '—'

def kvc_cell(r):
    if pd.isna(r['kvc_rb_layers']): return '未启用'
    return f"{int(r['kvc_rb_layers'])}层" if r['kvc_rb_layers'] else '—'

def share(r, col):
    if not r['lin_total']: return None
    return round(100*r[col]/r['lin_total'], 2)

import re as _re
def fam(m):
    """模型分组键：去掉尺寸(-32B/-235B-A22B)、微调(-Instruct/-Thinking/-it)、日期(-0905)后缀"""
    s = _re.sub(r'-\d{4}$', '', str(m))
    s = _re.sub(r'-(Instruct|Thinking|it)$', '', s)
    s = _re.sub(r'-\d+(\.\d+)?B(-A\d+(\.\d+)?B)?$', '', s)
    return s

def act_cell(r, c):
    v = r.get(c)
    if v == 'N/A': return 'N/A'
    return v if v else '—'

# ===== 表4 新口径：量化方案 + 回退占比（权重/激活合并，同一算子同一dtype） =====
def mode_cell(r):
    m = f"w{r['w']}a{r['a']}" if r['w'] else '未标注'
    if r['kvc']: m += '·c8'
    if pd.notna(r['fa3_fp_layers']): m += '·fa'
    return m

def scheme_cell(v):
    if v == 'N/A': return 'N/A'
    if isinstance(v, float) and np.isnan(v): return 'N/A'
    return v if v else '—'

def planned_cat_cell(r, cat):
    """量化方案 = 原计划方案：label 主位宽对应的 dtype（而非实际应用的 dtype）。
    实际全部升位宽时回退列显示实际 dtype，本列显示原计划，两者自然不同。"""
    w = r['w']
    applied = r[f'{cat}_dtypes']
    if applied == 'N/A' or (isinstance(applied, float) and np.isnan(applied)): return 'N/A'
    if w and w >= 16: return '不量化（FP16）'
    onplan = [d for d in str(applied).split('+') if d in BITS_D and BITS_D[d] <= (w or 16)] if applied and applied != 'FP（未量化）' else []
    if onplan: return '+'.join(onplan)
    plan = r.get('plan_dtypes')
    if plan and plan != 'N/A': return f"{plan}（计划）"
    return f"{int(w)}bit（计划）" if w else '—'

def planned_fa_cell(r):
    v = r.get('fa_dtypes')
    if v is None or (isinstance(v, float) and np.isnan(v)): return '未启用'
    if v == 'N/A': return 'N/A'
    fb = r.get('fa_bit')
    if fb is None or (isinstance(fb, float) and np.isnan(fb)): return v  # 无 fa 主位宽标注，方案即实际 dtype
    onplan = [d for d in str(v).split('+') if d in BITS_D and BITS_D[d] <= fb]
    if onplan: return '+'.join(onplan)
    plan = r.get('fa_plan_dtypes')
    if plan and plan != 'N/A': return f"{plan}（计划）"
    return f"{fb}bit（计划）"

def cat_parts(r, cat):
    """返回 [(kind, params或None, layers, dtype或None)]，kind ∈ FP/UP；N/A 返回 None"""
    fp_l = r[f'{cat}_fp_layers']
    if fp_l is None or (isinstance(fp_l, float) and np.isnan(fp_l)): return None
    fp_p = r[f'{cat}_fp_params']
    parts = []
    if fp_l:
        parts.append(('FP', fp_p if isinstance(fp_p, (int, float)) else None, int(fp_l), None))
    for (dt, nl, pp) in (r[f'{cat}_up_list'] or []):
        parts.append(('UP', pp if pp else None, nl, dt))
    return parts

def render_parts(parts, tot, tag=''):
    """kind/params/layers/dtype → 单元格文本；tag 用于标注线性注意力等子结构"""
    out = []
    for kind, pp, nl, dt in parts:
        t = f'{tag}' if tag else ''
        if kind == 'FP':
            if tot and pp is not None: out.append(f"FP:{pct1(100*pp/tot)}%（{pp/1e9:.2f}B/{nl}层）")
            else: out.append(f"FP:{t}{nl}层（参数N/A）")
        else:
            if tot and pp: out.append(f"升位宽:{pct1(100*pp/tot)}%→{dt}（{pp/1e9:.2f}B/{nl}层）")
            else: out.append(f"升位宽:{t}{nl}层→{dt}（参数N/A）")
    return out

def uniform_outcome(parts, tot):
    """整类回退 100% 且结果单一时，返回简化显示（方案, 回退列='—'）；否则 None"""
    if not parts: return None
    kinds = {p[0] for p in parts}
    unk = any(p[1] is None for p in parts)
    ps = sum(p[1] or 0 for p in parts)
    if kinds == {'FP'}:
        if tot and not unk and ps >= tot * 0.999: return ('不量化（FP16）', '—')
        return None
    if kinds == {'UP'}:
        dts = {p[3] for p in parts}
        if len(dts) == 1 and tot and not unk and ps >= tot * 0.999: return (dts.pop(), '—')
    return None

def majority_split(parts, tot):
    """回退项合计≈100%（计划方案完全未落地）且某一项严格过半时：
    多数派作为实际量化方案，其余项仍列为回退。返回 (方案文本, 少数派parts) 或 None"""
    if not parts or not tot: return None
    if any(p[1] is None for p in parts): return None
    ps = sum(p[1] for p in parts)
    if ps < tot * 0.999: return None          # 存在按计划正常执行的部分，方案列保持原计划
    agg = {}
    for kind, pp, nl, dt in parts:
        a = agg.setdefault((kind, dt), [0, 0]); a[0] += pp; a[1] += nl
    (bk, bd), (bp, bl) = max(agg.items(), key=lambda kv: kv[1][0])
    if len(agg) > 1 and bp <= ps / 2: return None   # 必须严格过半，50/50 保持原样
    minor = [(k, pp, nl, dt) for (k, dt), (pp, nl) in agg.items() if (k, dt) != (bk, bd)]
    scheme = '不量化（FP16）' if bk == 'FP' else bd
    return scheme, minor

def cat_share_cell(r, cat):
    """类别回退：占比 + 参数量 + 层数（分母=该类别参数总量）"""
    parts = cat_parts(r, cat)
    if parts is None: return 'N/A'
    out = render_parts(parts, r[f'{cat}_tot'])
    return '；'.join(out) if out else '—'

def attn_share_cell(r, parts=None):
    """attn 列 = 全注意力 + 线性注意力；parts 可覆盖（多数派规则时只渲染少数派）"""
    if parts is None: parts = cat_parts(r, 'attn')
    if parts is None: return 'N/A'
    out = render_parts(parts, r['attn_tot'])
    la_fp = r.get('la_fp_layers') or 0
    if la_fp and not (isinstance(la_fp, float) and np.isnan(la_fp)):
        pp = r.get('la_fp_params')
        ptxt = f"{pp/1e9:.2f}B" if pp else "参数N/A"
        out.append(f"FP:线性{int(la_fp)}层（{ptxt}）")
    for item in (r.get('la_up_list') or []):
        dt, nl = item[0], item[1]; par = item[2] if len(item) > 2 else 0
        ptxt = f"{par/1e9:.2f}B" if par else "参数N/A"
        out.append(f"升位宽:线性{nl}层→{dt}（{ptxt}）")
    return '；'.join(out) if out else '—'

def mlp_share_cell(r, parts=None, tot=None):
    """mlp 列 = dense MLP + 路由专家 + 共享专家，合计参数总量作分母；parts 可覆盖（多数派规则）"""
    pm = cat_parts(r, 'mlp'); pe = cat_parts(r, 'moe'); ps = cat_parts(r, 'shared')
    if pm is None and pe is None and ps is None: return 'N/A'
    if tot is None:
        tot = (r['mlp_tot'] or 0) + (r['moe_tot'] or 0) + (r['shared_tot'] or 0)
        if tot == 0: tot = None
    if parts is None:
        agg = {}
        for kind, pp, nl, dt in (pm or []) + (pe or []) + (ps or []):
            a = agg.setdefault((kind, dt), [0, 0, False])   # [params, layers, 存在未知参数]
            a[1] += nl
            if pp is None: a[2] = True
            else: a[0] += pp
        parts = []
        for (kind, dt), (pp, nl, unk) in sorted(agg.items()):
            parts.append((kind, None if (unk and pp == 0) else pp, nl, dt))
    out = render_parts(parts, tot)
    return '；'.join(out) if out else '—'

def merged_scheme_cell(r, cats):
    """合并类别的计划方案：各类别 on-plan dtype 的并集；都没有则回退到配置级计划 dtype"""
    w = r['w']
    if w and w >= 16: return '不量化（FP16）'
    applied = set()
    for c in cats:
        v = r[f'{c}_dtypes']
        if v and v != 'N/A' and v != 'FP（未量化）' and not (isinstance(v, float) and np.isnan(v)):
            applied |= set(str(v).split('+'))
    onplan = sorted([d for d in applied if d in BITS_D and BITS_D[d] <= (w or 16)])
    if onplan: return '+'.join(onplan)
    plan = r.get('plan_dtypes')
    if plan and plan != 'N/A' and not (isinstance(plan, float) and np.isnan(plan)): return f"{plan}（计划）"
    if not applied and any(r[f'{c}_dtypes'] == 'N/A' for c in cats): return 'N/A'
    return f"{int(w)}bit（计划）" if w else '—'

def fa_scheme_cell(r):
    v = r.get('fa_dtypes')
    if v is None or (isinstance(v, float) and np.isnan(v)): return '未启用'
    return scheme_cell(v)

def fa_share_cell(r):
    """fa 回退占比（按层数统计，分母=总层数L）"""
    if str(r.get('fa_dtypes')) == 'N/A': return 'N/A'
    if pd.isna(r['fa3_fp_layers']): return '未启用'
    L = r['L'] or 0
    parts = []
    if r['fa3_fp_layers']:
        parts.append(f"FP:{pct1(100*r['fa3_fp_layers']/L)}%（{int(r['fa3_fp_layers'])}/{L}层）" if L else f"FP:{int(r['fa3_fp_layers'])}层")
    for (dt, nl) in (r.get('fa_up_list') or []):
        parts.append(f"升位宽:{pct1(100*nl/L)}%→{dt}（{nl}/{L}层）" if L else f"升位宽:{nl}层→{dt}")
    return '；'.join(parts) if parts else '—'

def kvc_scheme_cell(r):
    if not r['kvc']: return '未启用'
    if r['L'] == 0: return 'N/A'
    return r.get('kvc_dtype') or 'int8'

def kvc_share_cell(r):
    if not r['kvc']: return '—'
    if pd.isna(r['kvc_rb_layers']): return 'N/A'
    rb = r['kvc_rb_layers']; L = r['L'] or 0
    if not rb: return '—'
    return f"FP:{pct1(100*rb/L)}%（{int(rb)}/{L}层）" if L else f"FP:{int(rb)}层"

# ===== 统一排序：合计回退计算量占比降序，无法核算的行排最后 =====
na_mask = df['lin_total'].isna() | (df['lin_total']==0)
df_rank = df[~na_mask].copy()
df_rank['fp_share']=df_rank.apply(lambda r: share(r,'fp_par_all'),axis=1)
df_rank['up_share']=df_rank.apply(lambda r: share(r,'up_par_all'),axis=1)
# 目标位宽即 w16a16（如纯稀疏配置）：不在位宽量化回退口径内，FP 不计为回退（显示 —、不进 Top15/均值）
_fp16t = (df_rank['w']>=16)&(df_rank['a']>=16)
df_rank.loc[_fp16t,'fp_share']=np.nan
df_rank['tot_share']=df_rank['fp_share']+df_rank['up_share']
df_rank=df_rank.sort_values(['tot_share','yaml'],ascending=[False,True])
df_ord=pd.concat([df_rank, df[na_mask].sort_values('yaml')]).reset_index(drop=True)
df_ord.insert(0,'seq',range(1,len(df_ord)+1))
dperf=df_rank  # 统计用
# 100% 回退 = 整模型不量化（如纯稀疏配置），属预期行为而非回退，不计入回退统计
dperf_rb=dperf[dperf['fp_share']<99.9]

def n_fp(c): return int((df[c].fillna(0)>0).sum())
def has_up(c): return int(df[c].fillna('').astype(str).str.strip().ne('').sum())
def n_fp_act(c): return int((df[c].fillna('').astype(str).str.startswith('FP')).sum())
def n_up_act(c): return int((df[c].fillna('').astype(str).str.contains('升位宽')).sum())
fa3_en=int(df.fa3_fp_layers.notna().sum()); kvc_en=int(df.kvc_rb_layers.notna().sum())
ny=df.yaml.nunique()

# ================= Excel =================
hdr_fill = PatternFill('solid', fgColor='305496')
hdr_font = Font(color='FFFFFF', bold=True, size=10)
thin = Border(*[Side(style='thin', color='BFBFBF')]*4)
wrap = Alignment(vertical='center', wrap_text=True)
def style_sheet(ws, widths, freeze='D2'):
    for c in range(1, ws.max_column+1):
        cell = ws.cell(1, c); cell.fill=hdr_fill; cell.font=hdr_font
        cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True); cell.border=thin
        ws.column_dimensions[get_column_letter(c)].width = widths[c-1] if c-1 < len(widths) else 14
    for r in range(2, ws.max_row+1):
        for c in range(1, ws.max_column+1):
            cell=ws.cell(r,c); cell.border=thin; cell.alignment=wrap
    ws.freeze_panes=freeze
    ws.auto_filter.ref = ws.dimensions

wb = Workbook()
ws = wb.active; ws.title='逐配置明细'
head = ['序号','YAML配置','适用模型','w位宽','a位宽','启用fa3','启用KVcache量化',
        '总层数','attn层数','MoE层数','dense MLP层数','MTP层数',
        'attn 权重FP回退层数','attn 权重FP回退参数量(B)','attn 权重升位宽回退',
        'moe 权重FP回退层数','moe 权重FP回退参数量(B)','moe 权重升位宽回退',
        'mlp 权重FP回退层数','mlp 权重FP回退参数量(B)','mlp 权重升位宽回退',
        'attn 激活回退','moe 激活回退','mlp 激活回退',
        'fa3 FP回退层数','fa3 FP回退层明细','fa3 升位宽回退',
        'KVcache回退层数','KVcache回退层明细','备注']
ws.append(head)
def il(v): return int(v) if pd.notna(v) and v is not None else v
for _,r in df_ord.iterrows():
    ws.append([r['seq'], r['yaml'], r['model'], r['w'], r['a'],
               '是' if pd.notna(r['fa3_fp_layers']) else '否', '是' if r['kvc'] else '否',
               r['L'], r['attn_layers_total'], r['moe_layers_total'], r['dense_layers_total'], r['mtp_layers'],
               il(r['attn_fp_layers']), par2b(r['attn_fp_params']), r['attn_up'] or '',
               il(r['moe_fp_layers']), par2b(r['moe_fp_params']), r['moe_up'] or '',
               il(r['mlp_fp_layers']), par2b(r['mlp_fp_params']), r['mlp_up'] or '',
               act_cell(r,'attn_act'), act_cell(r,'moe_act'), act_cell(r,'mlp_act'),
               il(r['fa3_fp_layers']) if pd.notna(r['fa3_fp_layers']) else '未启用',
               str(r['fa3_fp_detail']) if not isinstance(r['fa3_fp_detail'],str) else r['fa3_fp_detail'],
               r['fa3_up'] or '',
               il(r['kvc_rb_layers']) if pd.notna(r['kvc_rb_layers']) else '未启用',
               str(r['kvc_rb_detail']) if not isinstance(r['kvc_rb_detail'],str) else r['kvc_rb_detail'],
               (r['note'] or '') + ('；部分模块参数未知(未计入总量)' if r.get('unk') else '')])
style_sheet(ws, [6,34,20,7,7,8,10, 7,7,7,9,7, 10,12,22, 10,12,18, 10,12,18, 14,14,14, 9,22,14, 9,22,36])

# ---- 性能损失分析 sheet（与明细同序） ----
ws3 = wb.create_sheet('性能损失分析')
ws3.append(['序号','YAML配置','适用模型','目标w位宽','全模型线性层参数(B)','FP回退参数(B)','FP计算量占比(%)',
            '升位宽回退参数(B)','升位宽计算量占比(%)','等效权重位宽(bit)','等效激活位宽(bit)','位宽偏差(eq-目标)','数据完整'])
for _,r in df_ord.iterrows():
    if pd.isna(r['lin_total']) or not r['lin_total']:
        ws3.append([r['seq'], r['yaml'], r['model'], r['w'], None,None,None,None,None,None,None,None,'无公开config，无法核算'])
        continue
    ws3.append([r['seq'], r['yaml'], r['model'], r['w'], round(r['lin_total']/1e9,2),
                round(r['fp_par_all']/1e9,2), None if pd.isna(r['fp_share']) else r['fp_share'],
                round(r['up_par_all']/1e9,2), r['up_share'],
                round(r['eq_w'],2), round(r['eq_a'],2) if pd.notna(r.get('eq_a')) else None,
                round(r['eq_w']-r['w'],2) if r['w'] else None,
                ('目标即FP16，不计回退；' if pd.isna(r['fp_share']) else '')+('部分参数未知*' if r.get('unk') else '完整')])
style_sheet(ws3, [6,34,20,9,13,11,11, 13,13,11,11,11,14])

# ---- 汇总 ----
ws2 = wb.create_sheet('汇总')
ws2.append(['统计项','数值','说明'])
rows2 = [
 ('YAML配置总数', ny, f'共{len(df)}行（一个YAML可适配多个模型）'),
 ('attn 权重FP回退的配置行', int(((df['attn_fp_layers'].fillna(0)>0)|(df['la_fp_layers'].fillna(0)>0)).sum()), '含线性注意力；未被任何量化块覆盖，保持浮点'),
 ('attn 权重升位宽回退的配置行', int(((df['attn_up'].fillna('').astype(str).str.len()>0)|(df['la_up_list'].apply(len)>0)).sum()), '含线性注意力；量化位宽高于label.w_bit，注明目标类型'),
 ('attn 激活FP回退的配置行', int((df['attn_act'].fillna('').astype(str).str.contains('FP')|(df['la_act_fp_layers'].fillna(0)>0)).sum()), '含线性注意力；未覆盖时权重激活一起FP（a16配置不计）'),
 ('attn 激活升位宽回退的配置行', int((df['attn_act'].fillna('').astype(str).str.contains('升位宽')|(df['la_act_up_list'].apply(len)>0)).sum()), '含线性注意力；激活位宽高于label.a_bit'),
 ('mlp 权重FP回退的配置行', int(((df['mlp_fp_layers'].fillna(0)>0)|(df['moe_fp_layers'].fillna(0)>0)|(df['shared_fp_layers'].fillna(0)>0)).sum()), '含路由/共享专家并集'),
 ('mlp 权重升位宽回退的配置行', int(((df['mlp_up'].fillna('').astype(str).str.len()>0)|(df['moe_up'].fillna('').astype(str).str.len()>0)|(df['shared_up'].fillna('').astype(str).str.len()>0)).sum()), '含路由/共享专家并集'),
 ('mlp 激活FP回退的配置行', int((df['mlp_act'].fillna('').astype(str).str.contains('FP')|df['moe_act'].fillna('').astype(str).str.contains('FP')|df['shared_act'].fillna('').astype(str).str.contains('FP')).sum()), '含路由/共享专家并集'),
 ('mlp 激活升位宽回退的配置行', int((df['mlp_act'].fillna('').astype(str).str.contains('升位宽')|df['moe_act'].fillna('').astype(str).str.contains('升位宽')|df['shared_act'].fillna('').astype(str).str.contains('升位宽')).sum()), '含路由/共享专家并集'),
 ('启用fa3量化的配置行', fa3_en, 'fa3量化的是注意力算子的激活(非权重)'),
 ('fa3 有FP回退的配置行', n_fp('fa3_fp_layers'), '注意力算子未量化层'),
 ('fa3 有升位宽回退的配置行', has_up('fa3_up'), '高于label.fa_bit'),
 ('启用KVcache量化的配置行', kvc_en, ''),
 ('KVcache 有回退的配置行', n_fp('kvc_rb_layers'), ''),
 ('—— 性能损失口径 ——','',''),
 ('FP计算量占比 最大值', f"{dperf_rb['fp_share'].max():.2f}%", '回退模块参数/全模型线性层参数（≈每token GEMM计算量占比）；不含目标即FP16的配置行'),
 ('FP计算量占比 平均值', f"{dperf_rb['fp_share'].mean():.2f}%", f"剔除{len(dperf)-len(dperf_rb)}行目标即FP16的配置（纯稀疏等，不在位宽量化回退口径）后平均"),
 ('升位宽计算量占比 最大值', f"{dperf['up_share'].max():.2f}%", ''),
 ('升位宽计算量占比 平均值', f"{dperf['up_share'].mean():.2f}%", ''),
 ('等效权重位宽 范围', f"{dperf['eq_w'].min():.2f} ~ {dperf['eq_w'].max():.2f} bit", '按参数加权，FP按16bit计'),
]
for r in rows2: ws2.append(list(r))
for c in range(1,4):
    cell=ws2.cell(1,c); cell.fill=hdr_fill; cell.font=hdr_font; cell.border=thin
for w,c in zip([40,16,60],[1,2,3]): ws2.column_dimensions[get_column_letter(c)].width=w
for rr in range(2, ws2.max_row+1):
    for c in range(1,4): ws2.cell(rr,c).border=thin; ws2.cell(rr,c).alignment=wrap
os.makedirs(OUT, exist_ok=True)
wb.save(os.path.join(OUT, '最佳实践回退率统计_明细.xlsx'))
print('xlsx saved, perf rows:', len(dperf))

# ================= Markdown =================
stats=dict(attn_fp=n_fp('attn_fp_layers'),attn_up=has_up('attn_up'),
  moe_fp=n_fp('moe_fp_layers'),moe_up=has_up('moe_up'),
  mlp_fp=n_fp('mlp_fp_layers'),mlp_up=has_up('mlp_up'),
  fa3_en=fa3_en,fa3_fp=n_fp('fa3_fp_layers'),fa3_up=has_up('fa3_up'),
  kvc_en=kvc_en,kvc_fp=n_fp('kvc_rb_layers'))
L=[]
L.append('# msModelSlim 最佳实践（lab_practice）回退率与性能损失统计报告\n')
L.append(f"- 数据源：https://gitcode.com/Ascend/msmodelslim/tree/master/lab_practice （{ny} 个 YAML 最佳实践配置，展开为 {len(df)} 条 配置×模型 记录——一个 YAML 的 verified_model_types 声明多个适用模型时每个模型各占一行；default/ 目录 2 个无适用模型的兜底模板不计入）")
L.append('- 统计日期：2026-09-07\n')
L.append('## 一、统计口径\n')
L.append('1. **回退分两类，均记录**：')
L.append('   - **浮点回退（FP）**：模块未被任何量化配置覆盖，保持浮点（BF16/FP16）。')
L.append('   - **升位宽回退（→类型）**：模块被量化、但量化位宽高于该配置 label 标注的主位宽（如主位宽 w4 的模型中部分层量化为 int8/mxfp8），报告中**注明回退到的目标类型**。')
L.append('2. **层数口径**：按 Transformer 层/块计数。某层内任一该类别模块回退即计该层；参数量为该层实际回退模块的参数之和。')
L.append('3. **参数口径**：按对应模型公开 config.json 的结构参数计算，单位 B（十亿参数）。结构性模块（router/gate、lm_head、embed、norm、调制层 mod 等）不计入四类指标，仅在备注说明；DeepSeek-V4 注意力参数已按官方 inference/model.py 核算；仅无具体目标模型的通用模板（llm_transformers）标注 N/A。')
L.append('4. **attn linear**：q/k/v/o 投影（MLA 架构含 q_a/q_b/kv_a/kv_b_proj，DSA 含 indexer 的 wq_b/wk/weights_proj）**+ 线性注意力投影**（in_proj/out_proj 等，参数按官方建模代码核算）。**mlp linear**：dense 层 MLP 投影 **+ 路由专家 + 共享专家**（shared_experts）。**fa3**：注意力算子量化（fa3_quant）的层覆盖。**KVcache**：dynamic_cache 量化（c8 配置；GLM-5.2 等 knope 场景以 fa3 路径计）。')
L.append('5. V0 旧格式（modelslim_v0，calib_cfg.disable_names）按 disable 名单判定回退；Wan2.2-A14B 为高/低噪双专家模型，块数按单个 DiT 计。')
L.append('6. **权重侧与激活侧的区分**：①②③ 类别中，**权重与激活由同一个量化算子（同一份 qconfig）成对处理**，因此第四节以"量化方案"一列统一描述（dtype 即权重/激活的成对类型）；个别配置激活位宽与 label.a_bit 不一致的，在备注或 Excel 明细中单独标注。④ fa3 量化的是注意力算子的**激活**（Q/K/V），⑤ KVcache 是**缓存激活**——④⑤ 本身不属于权重。')
L.append('7. **性能损失口径（第三节）**：')
L.append('   - **计算量占比**：线性层每 token 的 GEMM 计算量 ≈ 2×参数量，故用"回退模块参数量 ÷ 全模型线性层参数总量"近似回退带来的额外计算占比（分母不含 router/embed/lm_head/norm/vision/mod；标注 * 的行存在参数未知的模块，实际占比略高）。')
L.append('   - **等效平均位宽**：全模型线性层按参数加权的平均位宽（浮点回退按 16bit 计），等效权重位宽与 label.w_bit 的差值反映权重压缩率/带宽损失；等效激活位宽同理对比 label.a_bit。GEMM 实际速度由权重和激活中较高的一方决定。')
L.append('8. **目标即 w16a16 的配置（如纯浮点稀疏）不做位宽量化，不属于回退统计口径**：其 FP 不计为回退，性能表与看板中 FP 占比显示 `—`，且不进入 Top15 排序与均值统计。\n')
L.append('## 二、总体结论\n')
L.append('| 指标 | 侧别 | 启用/适用配置行 | 浮点回退（FP） | 升位宽回退（注明目标类型） |')
L.append('|---|---|---|---|---|')
L.append(f"| ① attn linear（含线性注意力） | 权重 | {len(df)} | {int(((df['attn_fp_layers'].fillna(0)>0)|(df['la_fp_layers'].fillna(0)>0)).sum())} 行 | {int(((df['attn_up'].fillna('').astype(str).str.len()>0)|(df['la_up_list'].apply(len)>0)).sum())} 行 |")
L.append(f"| ① attn linear（含线性注意力） | 激活 | {len(df)} | {int((df['attn_act'].fillna('').astype(str).str.contains('FP')|(df['la_act_fp_layers'].fillna(0)>0)).sum())} 行 | {int((df['attn_act'].fillna('').astype(str).str.contains('升位宽')|(df['la_act_up_list'].apply(len)>0)).sum())} 行 |")
L.append(f"| ② mlp linear（含路由/共享专家） | 权重 | {len(df)} | {int(((df['mlp_fp_layers'].fillna(0)>0)|(df['moe_fp_layers'].fillna(0)>0)|(df['shared_fp_layers'].fillna(0)>0)).sum())} 行 | {int(((df['mlp_up'].fillna('').astype(str).str.len()>0)|(df['moe_up'].fillna('').astype(str).str.len()>0)|(df['shared_up'].fillna('').astype(str).str.len()>0)).sum())} 行 |")
L.append(f"| ② mlp linear（含路由/共享专家） | 激活 | {len(df)} | {int((df['mlp_act'].fillna('').astype(str).str.contains('FP')|df['moe_act'].fillna('').astype(str).str.contains('FP')|df['shared_act'].fillna('').astype(str).str.contains('FP')).sum())} 行 | {int((df['mlp_act'].fillna('').astype(str).str.contains('升位宽')|df['moe_act'].fillna('').astype(str).str.contains('升位宽')|df['shared_act'].fillna('').astype(str).str.contains('升位宽')).sum())} 行 |")
L.append(f"| ③ attn (fa3) | 激活（算子） | {stats['fa3_en']} 行启用 | {stats['fa3_fp']} 行 | {stats['fa3_up']} 行 |")
L.append(f"| ④ KVcache | 激活（缓存） | {stats['kvc_en']} 行启用 | {stats['kvc_fp']} 行 | — |")
L.append('')
L.append(f"性能损失维度：可核算的 {len(dperf)} 行中，FP 回退计算量占比最大 {dperf_rb['fp_share'].max():.2f}%、平均 {dperf_rb['fp_share'].mean():.2f}%（剔除 {len(dperf)-len(dperf_rb)} 行目标即 FP16 的配置——纯稀疏等不做位宽量化的配置不在回退口径内）；升位宽回退计算量占比最大 {dperf['up_share'].max():.2f}%、平均 {dperf['up_share'].mean():.2f}%；等效权重位宽范围 {dperf['eq_w'].min():.2f}~{dperf['eq_w'].max():.2f} bit。\n")
L.append('单元格书写格式：`FP:层数/参数量B`（浮点回退）；`升位宽:层数/参数量B→目标类型`（非浮点回退）；`—` 表示无回退；`N/A` 表示无公开 config 无法核算。\n')
L.append('> 注：权重与激活由同一量化算子成对处理，但回退判定各自对照 label 位宽（权重对 `w_bit`、激活对 `a_bit`），故两侧行数不必完全相等——完全未覆盖的模块权重与激活一起 FP、两侧均计（仅 a16 配置激活 FP16 属计划目标不计，差值即此类行）；升位宽侧差异来自 w4a8 配置：模块落到 8bit，对权重（目标4bit）是升位宽回退、对激活（目标8bit）是达标；fa3/KVcache 本身就是激活侧对象，无权重对应行。\n')
L.append('## 三、性能损失分析（计算量占比 + 等效平均位宽）\n')
L.append('**本表与第四节明细表同序排列（按"FP+升位宽"合计计算量占比降序，无法核算的行排最后），两行序号一一对应，方便对照。**\n')
L.append('| 序号 | 配置 | 模型 | 目标位宽 | 线性层总量(B) | FP回退参数(B) | FP占比(%) | 升位宽参数(B) | 升位宽占比(%) | 等效权重位宽 | 等效激活位宽 | 位宽偏差 |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
for _,r in df_ord.iterrows():
    if pd.isna(r['lin_total']) or not r['lin_total']:
        L.append(f"| {r['seq']} | {r['yaml']} | {r['model']} | {r['w']} | N/A | — | — | — | — | — | — | — |")
        continue
    if pd.isna(r['fp_share']):  # 目标即 w16a16：FP 不计回退
        L.append('| {} | {} | {} | {} | {:.2f} | — | — | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:+.2f} |'.format(
            r['seq'], r['yaml'], r['model'], r['w'], r['lin_total']/1e9,
            r['up_par_all']/1e9, r['up_share'], r['eq_w'], r['eq_a'], r['eq_w']-r['w']))
        continue
    L.append('| {} | {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:+.2f} |'.format(
        r['seq'], r['yaml'], r['model'], r['w'], r['lin_total']/1e9, r['fp_par_all']/1e9, r['fp_share'],
        r['up_par_all']/1e9, r['up_share'], r['eq_w'], r['eq_a'], r['eq_w']-r['w']))
L.append('')
L.append('## 四、逐配置明细（与第三节同序）\n')
L.append('**口径：attn 列 = 全注意力 + 线性注意力（线性注意力回退标注"线性"，层数与参数量均按官方建模代码核算：Qwen 系 GatedDeltaNet、Kimi/GLM 系 KDA、DeepSeek-V4 压缩注意力）；mlp 列 = dense MLP + 路由专家（合并计参数量）；共享专家仍只在备注标注。权重与激活由同一算子成对量化，故每类只列一个"量化方案"= 原计划方案（label 主位宽对应的 dtype，如 w4 配置计划用 int4/mxfp4）；若该类别实际全部升位宽/浮点回退，方案列显示计划 dtype 并标注（计划），与回退列的实际 dtype 自然不同。回退单元格格式：`类型:占比%（参数量B/层数）`，attn/mlp 按参数量统计（回退参数 ÷ 该类别参数总量）；fa 按层数统计（回退层 ÷ 总层数）；KVcache 同为按层数。`—` 表示无回退；`未启用` 表示未启用对应量化；若某类 100% 回退，则方案列直接显示实际结果（全 FP 显示`不量化（FP16）`；全部升到同一类型则方案列写该类型），回退列显示 `—`；回退项合计≈100% 且其中一项严格过半时（计划方案完全未落地），方案列显示占比过半的实际结果（FP 过半则显示`不量化（FP16）`），仅将其余项列为回退。**`N/A` 表示无公开 config 无法核算。**\n')
dash_rows = []
L.append('| 序号 | 模型 | 量化模式 | attn线性层量化方案 | attn线性层回退占比（按参数量） | mlp线性层量化方案 | mlp线性层回退占比（按参数量） | kvcache量化方案 | kvcache量化回退占比 | fa量化方案 | fa量化回退占比（按层数统计） | 备注 |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
for _,r in df_ord.iterrows():
    note=(r['note'] or '').replace('|','/')
    act_notes=[]
    for cat,label_zh in (('attn','attn'),('moe','mlp-专家'),('mlp','mlp'),('shared','共享专家')):
        av = r.get(f'{cat}_act')
        if not av or av=='N/A': continue
        wparts=set()
        fp_l = r[f'{cat}_fp_layers']
        if fp_l and not (isinstance(fp_l,float) and np.isnan(fp_l)): wparts.add(f"FP:{int(fp_l)}层")
        for seg in str(r[f'{cat}_up']).split('; '):
            if seg.strip(): wparts.add('升位宽:'+seg.replace('/0.00B→','→'))
        av_norm = str(av).replace('/0.00B→','→')
        if set(av_norm.split('; ')) - wparts:
            act_notes.append(f"{label_zh}激活:{av_norm}")
    if act_notes:
        note = (note+'；' if note else '') + '；'.join(act_notes)
    # 100% 回退的简化显示：全FP=不量化；全升到同一dtype=方案直接写该dtype
    ap = cat_parts(r, 'attn')
    la_fp = r.get('la_fp_layers') or 0
    la_parts = ([('FP', None, int(la_fp), None)] if la_fp and not (isinstance(la_fp,float) and np.isnan(la_fp)) else []) + [('UP', None, it[1], it[0]) for it in (r.get('la_up_list') or [])]
    tot_mlp = (r['mlp_tot'] or 0) + (r['moe_tot'] or 0) + (r['shared_tot'] or 0)
    mp = cat_parts(r, 'mlp'); me = cat_parts(r, 'moe'); ms = cat_parts(r, 'shared')
    attn_u = uniform_outcome((ap or []) + la_parts, r['attn_tot']) if ap is not None else None
    mlp_u = uniform_outcome((mp or []) + (me or []) + (ms or []), tot_mlp or None) if not (mp is None and me is None and ms is None) else None
    # 多数派规则：回退合计≈100% 时（计划方案完全未落地），占比过半的一方作为实际量化方案
    attn_m = majority_split((ap or []), r['attn_tot']) if (not attn_u and ap is not None) else None
    mlp_m = majority_split((mp or []) + (me or []) + (ms or []), tot_mlp or None) if (not mlp_u and not (mp is None and me is None and ms is None)) else None
    if attn_u: attn_scheme, attn_share = attn_u
    elif attn_m:
        attn_scheme, minor = attn_m
        attn_share = attn_share_cell(r, minor)      # 少数派 + 线性注意力片段
    else: attn_scheme, attn_share = planned_cat_cell(r,'attn'), attn_share_cell(r)
    if mlp_u: mlp_scheme, mlp_share = mlp_u
    elif mlp_m:
        mlp_scheme, minor = mlp_m
        mlp_share = mlp_share_cell(r, minor, tot_mlp or None)
    else: mlp_scheme, mlp_share = merged_scheme_cell(r,['mlp','moe','shared']), mlp_share_cell(r)
    # 结构性高回退的自动备注（避免误读为异常）
    if mlp_scheme == '不量化（FP16）' and attn_scheme != '不量化（FP16）' and '仅量化' not in note:
        note = (note+'；' if note else '') + '配置仅量化注意力，mlp整体浮点（预期设计）'
    if str(r['model']).startswith('DeepSeek-V4') and (r['attn_fp_params'] or 0) + sum(p for *_,p in (r.get('attn_up_list') or [])) > 0.3*(r['attn_tot'] or 1) and 'wo_a/wo_b' not in note:
        note = (note+'；' if note else '') + 'attn高回退源于配置刻意排除 wo_a/wo_b 输出投影及 DSA compressor/indexer（敏感模块），wq_a/wq_b/wkv 已量化'
    up_sh = 100*(r['up_par_all'] or 0)/r['lin_total'] if r['lin_total'] else 0
    if up_sh > 50 and '混合位宽' not in note:
        note = (note+'；' if note else '') + '配置本身为混合位宽策略：敏感模块按计划保持较高bit，升位宽属预期设计'
    L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
        r['seq'], r['model'], mode_cell(r),
        attn_scheme, attn_share, mlp_scheme, mlp_share,
        kvc_scheme_cell(r), kvc_share_cell(r),
        planned_fa_cell(r), fa_share_cell(r), note))
    dash_rows.append({'seq': int(r['seq']), 'model': r['model'], 'yaml': r['yaml'], 'family': fam(r['model']), 'mode': mode_cell(r),
        'attn_scheme': attn_scheme, 'attn_share': attn_share, 'mlp_scheme': mlp_scheme, 'mlp_share': mlp_share,
        'kvc_scheme': kvc_scheme_cell(r), 'kvc_share': kvc_share_cell(r),
        'fa_scheme': planned_fa_cell(r), 'fa_share': fa_share_cell(r), 'note': note,
        'lin_total': round((r['lin_total'] or 0)/1e9, 2) if r['lin_total'] else None,
        'fp_share': None if pd.isna(r['fp_share']) else round(float(r['fp_share']),2),
        'up_share': None if pd.isna(r['up_share']) else round(float(r['up_share']),2),
        'eq_w': round(r['eq_w'],2) if pd.notna(r['eq_w']) else None,
        'eq_a': round(r['eq_a'],2) if pd.notna(r['eq_a']) else None,
        'w': r['w'], 'a': r['a']})
# ---- 五、按模型分组 ----
L.append('## 五、按模型分组（同族模型聚合）\n')
L.append('分组键 = 模型名去掉尺寸（-32B / -235B-A22B）、微调（-Instruct/-Thinking/-it）与日期（-0905）后缀。组内按合计回退占比降序。\n')
L.append('| 模型分组 | 配置行数 | 模型 | FP回退占比(%) | 升位宽占比(%) |')
L.append('|---|---|---|---|---|')
def _rng(vals):
    vals = [v for v in vals if pd.notna(v)]
    if not vals: return '—'
    lo, hi = min(vals), max(vals)
    return f'{lo:.2f}' if abs(hi-lo) < 0.005 else f'{lo:.2f} ~ {hi:.2f}'
_g = {}
for _, r in df_ord.iterrows():
    if pd.isna(r['lin_total']) or not r['lin_total']: continue
    _g.setdefault(fam(r['model']), []).append(r)
_gsorted = sorted(_g.items(), key=lambda kv: -max((x['fp_share'] if pd.notna(x['fp_share']) else 0)+(x['up_share'] if pd.notna(x['up_share']) else 0) for x in kv[1]))
for gname, grows in _gsorted:
    models = '、'.join(dict.fromkeys(x['model'] for x in grows))
    L.append(f"| {gname} | {len(grows)} | {models} | {_rng([x['fp_share'] for x in grows])} | {_rng([x['up_share'] for x in grows])} |")
L.append('')
L.append('**规律**：同一模型的 FP 回退率在其各配置间基本相同——FP 回退是结构性的（同一模型的各 YAML 继承相同的 exclude/disable 名单：vision 塔、敏感投影、MTP 等完全不覆盖），与目标位宽无关；差异主要来自**升位宽**——同一排除集在 w4 目标下计为升位宽（int8>4bit），在 w8 目标下则达标，故 w4 配置升位宽占比显著高于同模型 w8 配置。\n')
open(os.path.join(OUT, '最佳实践回退率统计.md'), 'w').write('\n'.join(L))
# ---- 看板数据导出（供 gen_dashboard.py 使用） ----
import json as _json
_dash = {'rows': dash_rows,
         'summary': {'yamls': int(ny), 'rows': len(df), 'perf_rows': int(len(dperf)),
                     'fp_max': round(float(dperf_rb['fp_share'].max()),2), 'fp_avg': round(float(dperf_rb['fp_share'].mean()),2),
                     'up_max': round(float(dperf['up_share'].max()),2), 'up_avg': round(float(dperf['up_share'].mean()),2),
                     'eqw_min': round(float(dperf['eq_w'].min()),2), 'eqw_max': round(float(dperf['eq_w'].max()),2),
                     'eqw_avg': round(float(dperf['eq_w'].mean()),2),
                     'fa3_en': fa3_en, 'kvc_en': kvc_en,
                     'attn_fp_rows': stats['attn_fp'], 'mlp_fp_rows': stats['mlp_fp']}}
with open(os.path.join(WORK, 'dash_data.json'), 'w') as _f:
    _json.dump(_dash, _f, ensure_ascii=False)
print('md saved')
