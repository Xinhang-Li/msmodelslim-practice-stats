# -*- coding: utf-8 -*-
"""msModelSlim lab_practice 回退率 + 性能损失分析引擎"""
import os, re, json, glob, fnmatch
import yaml as _yaml

BITS = {'int4':4,'mxfp4':4,'fp4':4,'int8':8,'mxfp8':8,'fp8_e4m3':8,'fp8':8,'bfloat16':16,'float16':16}
LIN_TYPES = {'linear_quant','autoround_quant','trainable_linear_quant'}

# ---------------- YAML 解析 ----------------
def load_yamls(root):
    data = {}
    for p in sorted(glob.glob(os.path.join(root,'**','*.yaml'), recursive=True)):
        try:
            with open(p) as f: data[p] = _yaml.safe_load(f)
        except Exception: pass
    return data

def get_model_types(cfg):
    md = cfg.get('metadata',{}) or {}
    names = list((md.get('verified_tags',{}) or {}).keys())
    for x in (md.get('verified_model_types') or []):
        if x not in names: names.append(x)
    if not names:
        # 个别配置无 verified 标注，按 config_id/注释推断适用模型
        hint = {'glm_5_next_convert_mxfp8': ['GLM-5.3-Flash']}
        cid = str(md.get('config_id',''))
        if cid in hint: names = hint[cid]
    # DiT 静态表别名规范化（如 Wan2_1/Wan2.1 指向同一模型），避免重复行
    out=[]
    for n in names:
        nn = n if n in DIT_CFG else (n.replace('_','.') if n.replace('_','.') in DIT_CFG else n)
        if nn not in out: out.append(nn)
    return out

def brace_expand(p):
    m = re.search(r'\{([^{}]+)\}', p)
    if not m: return [p]
    out=[]
    for opt in m.group(1).split(','):
        out += brace_expand(p[:m.start()]+opt+p[m.end():])
    return out

def norm_name(n):
    for pre in ('language_model.model.','model.language_model.','thinker.model.','model.model.'):
        if n.startswith(pre): n = n[len(pre):]
    n = n.replace('block_sparse_moe','mlp')
    n = re.sub(r'\.ffn\.', '.mlp.', n)
    n = n.replace('moe.experts','mlp.experts')
    n = n.replace('shared_mlp','shared_experts')
    return n

def match_any(pats, name):
    variants = [name]
    if name.startswith('model.'): variants.append(name[len('model.'):])
    for p in pats or []:
        for pp in brace_expand(norm_name(str(p))):
            for nm in variants:
                if fnmatch.fnmatchcase(nm, pp): return True
                if pp.startswith('*') and pp.endswith('*') and len(pp)>2 and pp.strip('*') in nm: return True
    return False

def layers_in_patterns(pats):
    s=set()
    for p in pats or []:
        for m in re.finditer(r'(?:model\.layers|blocks|double_blocks|single_blocks|transformer_blocks)\.(\d+)', str(p)):
            s.add(int(m.group(1)))
    return s

def get_blocks(cfg):
    blocks=[]
    def add(b, override_type=None):
        b = b or {}
        w=((b.get('qconfig') or {}).get('weight') or {}).get('dtype')
        ad=((b.get('qconfig') or {}).get('act') or {}).get('dtype')
        fd=(b.get('qconfig') or {}).get('dtype')
        if fd is None:
            fd = (((b.get('details') or {}).get('fa_q') or {}).get('dtype'))
        t = override_type or b.get('type')
        inc = b.get('include') or (['*'] if t in LIN_TYPES else [])
        blocks.append((t, w, inc, b.get('exclude') or [], fd, ad))
    spec = cfg.get('spec',{}) or {}
    procs = list(spec.get('process') or [])
    for v in (spec.get('per_expert') or {}).values():
        procs += list(v or [])
    for proc in procs:
        proc = proc or {}
        if proc.get('type')=='group':
            for c in proc.get('configs',[]) or []: add(c)
        elif proc.get('strategies'):
            for c in proc['strategies']: add(c, proc.get('type'))
        else: add(proc)
    # modelslim_convert 格式：spec.linears 按 target 做离线权重转换/重排
    for entry in (spec.get('linears') or []):
        entry = entry or {}
        tgt = str(entry.get('target','')).upper()
        wd = ('mxfp8' if 'MXFP8' in tgt else 'mxfp4' if 'MXFP4' in tgt else
              'int8' if 'W8' in tgt else 'int4' if 'W4' in tgt else 'int8')
        blocks.append(('linear_quant', wd, entry.get('match') or [], [], None, wd))
    return blocks

# ---------------- 模型结构 ----------------
def _first(c, keys, default=None):
    for k in keys:
        if c.get(k) is not None: return c[k]
    return default

def mspec_from_config(mt, c):
    h = c.get('hidden_size'); L = c.get('num_hidden_layers') or c.get('num_layers') or 0
    nh = c.get('num_attention_heads') or 0
    nkvh = c.get('num_key_value_heads') or nh
    hd = c.get('head_dim') or (h//nh if nh else 0)
    kv_lora = c.get('kv_lora_rank') or 0
    q_lora = c.get('q_lora_rank') or 0
    nope = c.get('qk_nope_head_dim') or hd
    rope = c.get('qk_rope_head_dim') or 0
    vhd = c.get('v_head_dim') or hd
    idx_nh = c.get('index_n_heads') or 0
    idx_hd = c.get('index_head_dim') or 0
    n_exp = _first(c, ['n_routed_experts','num_experts','num_local_experts','moe_num_experts'], 0) or 0
    moe_inter = c.get('moe_intermediate_size') or c.get('expert_ffn_hidden_size') or (c.get('intermediate_size') if _first(c,['n_routed_experts','num_experts','num_local_experts','moe_num_experts'],0) else 0) or 0
    inter = c.get('intermediate_size') or c.get('ffn_hidden_size') or 0
    if c.get('dense_intermediate_size'): inter = c['dense_intermediate_size']
    n_shared = _first(c, ['n_shared_experts','num_shared_experts','shared_expert_num'], 0) or 0
    shared_inter = c.get('shared_expert_intermediate_size') or c.get('shared_intermediate_size') or (moe_inter*n_shared if moe_inter else 0)
    mtp = c.get('num_nextn_predict_layers') or 0
    exp_h = c.get('routed_expert_hidden_size') or 0   # LatentMoE：专家在潜空间工作（Kimi-K3）
    # MoE 层判定
    if n_exp:
        moe_layers = set(range(L))
        fk = _first(c, ['first_k_dense_replace','first_k_dense_layers'], None)
        if fk: moe_layers = set(range(fk, L))
        freq = c.get('moe_layer_freq')
        if isinstance(freq,list):
            moe_layers = {i for i,v in enumerate(freq) if v}
        elif freq and isinstance(freq,int) and freq>1:
            moe_layers = {i for i in moe_layers if (i+1)%freq==0}
        enum = c.get('moe_layers_enum')
        if isinstance(enum,str): enum=[int(x) for x in enum.split(',') if x.strip()]
        if enum: moe_layers = set(enum)
        mlt = c.get('mlp_layer_types')
        if mlt: moe_layers = {i for i,t in enumerate(mlt) if 'moe' in str(t).lower() or 'sparse' in str(t).lower()}
    else:
        moe_layers = set()
    dense_layers = set(range(L)) - moe_layers
    # 注意力层判定（混合线性注意力）
    attn_layers = set(range(L))
    la = c.get('linear_attn_config') or {}
    if la.get('full_attn_layers') is not None:
        attn_layers = {i for i in la['full_attn_layers'] if i < L}
    elif c.get('full_attention_interval'):
        iv = c['full_attention_interval']
        attn_layers = {i for i in range(L) if (i+1)%iv==0}
    # 线性注意力每层参数（输入投影桶 / 输出投影桶），结构尺寸取自模型 config 与官方建模代码
    la_in_p = la_out_p = None
    if h and c.get('linear_key_head_dim'):
        # Qwen 系 GatedDeltaNet：in_proj_qkv(z)=h→2*key+2*val，in_proj_ba/b+a=h→2*nv，out_proj=val→h
        kd = c['linear_key_head_dim']; nk_l = c.get('linear_num_key_heads') or 0
        vd = c.get('linear_value_head_dim') or kd; nv_l = c.get('linear_num_value_heads') or nk_l
        key_dim, val_dim = kd*nk_l, vd*nv_l
        la_in_p = h*(2*key_dim + 2*val_dim) + h*2*nv_l
        la_out_p = val_dim*h
    elif h and la.get('num_heads') and la.get('head_dim'):
        nhl, hdl = la['num_heads'], la['head_dim']; p = nhl*hdl
        if str(c.get('model_type',''))=='kimi_linear':
            # Kimi KDA：q/k/v + f_a/f_b(低秩) + b_proj + g_proj(use_full_rank_gate) 或 g_a/g_b
            gate = h*p if la.get('use_full_rank_gate') else (h*hdl + hdl*p)
            la_in_p = 3*h*p + h*hdl + hdl*p + h*nhl + gate
        else:
            # GLM-5.3(glm5_next) KDA：q/k/v + forget_gate.f_a/f_b(低秩) + b_proj + g_a/g_b(低秩)
            la_in_p = 3*h*p + 2*(h*hdl + hdl*p) + h*nhl
        la_out_p = p*h
    return dict(mt=mt, model_type=c.get('model_type',''), L=L, h=h, nh=nh, nkvh=nkvh, hd=hd,
                kv_lora=kv_lora, q_lora=q_lora, nope=nope, rope=rope, vhd=vhd,
                idx_nh=idx_nh, idx_hd=idx_hd, n_exp=n_exp, moe_inter=moe_inter, inter=inter,
                n_shared=n_shared, shared_inter=shared_inter,
                attn_layers=sorted(attn_layers), moe_layers=sorted(moe_layers),
                dense_layers=sorted(dense_layers), mtp=mtp,
                la_in_p=la_in_p, la_out_p=la_out_p, exp_h=exp_h,
                o_lora=c.get('o_lora_rank') or 0, o_groups=c.get('o_groups') or 0,
                cratios=c.get('compress_ratios'),
                gemma4=str(c.get('model_type','')).startswith('gemma4'),
                attn_types=c.get('layer_types') if str(c.get('model_type','')).startswith('gemma4') else None,
                k_eq_v=bool(c.get('attention_k_eq_v')),
                has_vision=bool(c.get('_has_vision') or c.get('vision_config') or c.get('visual')))

# DiT 模型（静态表 + 抓取修正）
DIT_CFG = {
 'Wan2.2-T2V-A14B': dict(L=40, h=5120, inter=13824),
 'Wan2.2-I2V-A14B': dict(L=40, h=5120, inter=13824),
 'Wan2.2-TI2V-5B':  dict(L=30, h=3072, inter=14336),
 'Wan2.1':          dict(L=40, h=5120, inter=13824),
 'FLUX.1-dev':      dict(L=57, h=3072, inter=12288),
 'Qwen-Image-Edit-2509': dict(L=60, h=3584, inter=14336),
 'HunyuanVideo':    dict(L=60, h=3072, inter=12288),
}

def mspec(mt, found):
    if mt in DIT_CFG:
        d = DIT_CFG[mt]
        return dict(mt=mt, dit=True, L=d['L'], h=d['h'], inter=d['inter'],
                    model_type='DiT', n_exp=0, attn_layers=list(range(d['L'])),
                    moe_layers=[], dense_layers=list(range(d['L'])), mtp=0, has_vision=False)
    c = found.get(mt)
    if c: return mspec_from_config(mt, c)
    return None

# ---------------- 每层模块清单 ----------------
def layer_modules(s, i):
    if s.get('dit'):
        d, f = s['h'], s['inter']
        mods={}
        for nm,pp in (('self_attn',4*d*d),('cross_attn',4*d*d)):
            mods[nm]=(pp,'attn',[nm])
        mods['ffn.0']=(d*f,'dense_mlp',['ffn.0'])
        mods['ffn.2']=(d*f,'dense_mlp',['ffn.2'])
        mt = s.get('mt','')
        if mt.startswith('HunyuanVideo'): ma=['txt_mod','img_mod']
        elif mt.startswith('Qwen-Image'): ma=['txt_mlp.net.2','img_mod.1','txt_mod.1']
        else: ma=['mod']
        mods['mod']=(None,'mod',ma)
        return mods
    h,nh,nkvh,hd = s['h'],s['nh'],s['nkvh'],s['hd']
    mods={}
    if i in s['attn_layers']:
        if s['kv_lora']>0:
            if s['q_lora']>0:
                mods['self_attn.q_a_proj']=(h*s['q_lora'],'attn',['self_attn.q_a_proj'])
                mods['self_attn.q_b_proj']=(s['q_lora']*nh*(s['nope']+s['rope']),'attn',['self_attn.q_b_proj'])
            else:
                mods['self_attn.q_proj']=(h*nh*(s['nope']+s['rope']),'attn',['self_attn.q_proj'])
            mods['self_attn.kv_a_proj_with_mqa']=(h*(s['kv_lora']+s['rope']),'attn',['self_attn.kv_a_proj_with_mqa'])
            mods['self_attn.kv_b_proj']=(s['kv_lora']*nh*(s['nope']+s['vhd']),'attn',['self_attn.kv_b_proj'])
            mods['self_attn.o_proj']=(nh*s['vhd']*h,'attn',['self_attn.o_proj'])
            if s['idx_nh']>0:
                mods['self_attn.indexer.wq_b']=(h*s['idx_nh']*s['idx_hd'],'attn',['self_attn.indexer.wq_b'])
                mods['self_attn.indexer.wk']=(h*s['idx_hd'],'attn',['self_attn.indexer.wk'])
                mods['self_attn.indexer.weights_proj']=(h*s['idx_nh'],'attn',['self_attn.indexer.weights_proj'])
        elif s['model_type']=='deepseek_v4':
            # 官方 inference/model.py：低秩Q + MQA(kv单头h→hd) + 分组低秩O + 逐层Compressor/Indexer
            ql = s['q_lora']; ol = s.get('o_lora') or 1024; og = s.get('o_groups') or 8
            mods['self_attn.wq_a']=(h*ql,'attn',['self_attn.wq_a'])
            mods['self_attn.wq_b']=(ql*nh*hd,'attn',['self_attn.wq_b'])
            mods['self_attn.wkv']=(h*hd,'attn',['self_attn.wkv'])
            mods['self_attn.wo_a']=((nh*hd//og)*(og*ol),'attn',['self_attn.wo_a'])
            mods['self_attn.wo_b']=(og*ol*h,'attn',['self_attn.wo_b'])
            crl = s.get('cratios') or []
            cr = crl[i] if i < len(crl) else 0
            if cr:
                coff = 2 if cr==4 else 1   # ratio==4 为重叠窗口（coff=2），128 为非重叠
                mods['self_attn.compressor.wkv']=(h*coff*hd,'attn',['self_attn.compressor.wkv'])
                mods['self_attn.compressor.wgate']=(h*coff*hd,'attn',['self_attn.compressor.wgate'])
                if cr==4:   # Indexer 仅存在于 ratio==4 的层
                    mods['self_attn.indexer.wq_b']=(ql*s['idx_nh']*s['idx_hd'],'attn',['self_attn.indexer.wq_b'])
                    mods['self_attn.indexer.weights_proj']=(h*s['idx_nh'],'attn',['self_attn.indexer.weights_proj'])
                    mods['self_attn.indexer.compressor.wkv']=(h*2*s['idx_hd'],'attn',['self_attn.indexer.compressor.wkv'])
                    mods['self_attn.indexer.compressor.wgate']=(h*2*s['idx_hd'],'attn',['self_attn.indexer.compressor.wgate'])
        else:
            mods['self_attn.q_proj']=(h*nh*hd,'attn',['self_attn.q_proj'])
            mods['self_attn.k_proj']=(h*nkvh*hd,'attn',['self_attn.k_proj'])
            # gemma4 全注意力层 K=V 共享，无独立 v_proj
            if not (s.get('k_eq_v') and s.get('attn_types') and s['attn_types'][i]=='full_attention'):
                mods['self_attn.v_proj']=(h*nkvh*hd,'attn',['self_attn.v_proj'])
            mods['self_attn.o_proj']=(nh*hd*h,'attn',['self_attn.o_proj'])
    else:
        # 线性注意力层：输入投影与输出投影分开统计（存在只量化输入投影的配置）
        mods['linear_attn.in_proj']=(s.get('la_in_p'),'linear_attn',[
            'linear_attn.in_proj_qkvz','linear_attn.in_proj_qkv','linear_attn.in_proj_z',
            'linear_attn.in_proj_ba','linear_attn.in_proj_b','linear_attn.in_proj_a',
            'linear_attn.q_proj','linear_attn.k_proj','linear_attn.v_proj',
            'linear_attn.g_proj','linear_attn.g_a_proj','linear_attn.g_b_proj',
            'linear_attn.f_a_proj','linear_attn.f_b_proj','linear_attn.b_proj',
            'linear_attn.forget_gate.f_a_proj','linear_attn.forget_gate.f_b_proj'])
        mods['linear_attn.out_proj']=(s.get('la_out_p'),'linear_attn',['linear_attn.out_proj','linear_attn.o_proj'])
    if s['n_exp']>0 and i in s['moe_layers']:
        exp_h = s.get('exp_h') or 0
        eh = exp_h if exp_h else h      # LatentMoE：专家输入/输出在潜空间（Kimi-K3: 3584）
        ep = eh*s['moe_inter']*s['n_exp'] if s['moe_inter'] else None
        if s.get('gemma4'):
            # gemma4：专家为 layers.experts.*（3D融合权重，适配器拆分后仍是 experts.* 前缀，不在 mlp 下）；
            # MoE 层的 mlp.* 即共享专家
            mods['experts.gate']=(ep,'moe',['experts.gate_proj','experts.gate_up_proj'])
            mods['experts.up']  =(ep,'moe',['experts.up_proj','experts.gate_up_proj'])
            mods['experts.down']=(ep,'moe',['experts.down_proj'])
            mods['mlp.gate_proj']=(h*s['inter'],'shared',['mlp.gate_proj'])
            mods['mlp.up_proj']  =(h*s['inter'],'shared',['mlp.up_proj'])
            mods['mlp.down_proj']=(h*s['inter'],'shared',['mlp.down_proj'])
        else:
            mods['mlp.experts.gate']=(ep,'moe',['mlp.experts.0.gate_proj','mlp.experts.0.w1','mlp.experts.gate_proj','mlp.experts.w1','ffn.experts.0.w1','ffn.experts.w1'])
            mods['mlp.experts.up']  =(ep,'moe',['mlp.experts.0.up_proj','mlp.experts.0.w3','mlp.experts.up_proj','mlp.experts.w3','ffn.experts.0.w3','ffn.experts.w3'])
            mods['mlp.experts.down']=(ep,'moe',['mlp.experts.0.down_proj','mlp.experts.0.w2','mlp.experts.down_proj','mlp.experts.w2','ffn.experts.0.w2','ffn.experts.w2'])
            mods['mlp.gate']=(h*s['n_exp'],'router',['mlp.gate','mlp.router','ffn.gate'])
            if exp_h:
                # LatentMoE 逐层共享潜空间投影（h→exp_h / exp_h→h），名字含 experts、随专家一起被量化
                mods['mlp.routed_expert_down_proj']=(h*exp_h,'moe',['mlp.routed_expert_down_proj'])
                mods['mlp.routed_expert_up_proj']=(exp_h*h,'moe',['mlp.routed_expert_up_proj'])
            if s['n_shared']>0 and s['shared_inter']:
                sp = 3*h*s['shared_inter']
                mods['mlp.shared_experts']=(sp,'shared',['mlp.shared_experts.gate_proj','mlp.shared_experts.up_proj',
                                                         'mlp.shared_experts.down_proj','mlp.shared_experts','ffn.shared_experts'])
    if i in s['dense_layers'] and s['inter']:
        mods['mlp.gate_proj']=(h*s['inter'],'dense_mlp',['mlp.gate_proj','mlp.w1'])
        mods['mlp.up_proj']  =(h*s['inter'],'dense_mlp',['mlp.up_proj','mlp.w3'])
        mods['mlp.down_proj']=(h*s['inter'],'dense_mlp',['mlp.down_proj','mlp.w2'])
    return mods

# ---------------- 主分析 ----------------
def analyze(yaml_path, data, found):
    cfg = data[yaml_path]
    md = cfg.get('metadata',{}) or {}
    label = md.get('label',{}) or {}
    kvc_label = bool(label.get('kv_cache', False))
    primary_w = label.get('w_bit')
    primary_fa = label.get('fa_bit')
    blocks = get_blocks(cfg)
    raw = open(yaml_path).read()
    lin_blocks = [b for b in blocks if b[0] in LIN_TYPES]
    fa3_blocks = [b for b in blocks if b[0]=='fa3_quant']
    cache_blocks = [b for b in blocks if b[0]=='dynamic_cache']
    legacy_dis = None
    spec_cfg = cfg.get('spec',{}) or {}
    if not lin_blocks and 'calib_cfg' in spec_cfg:
        cc = spec_cfg['calib_cfg'] or {}
        legacy_dis = cc.get('disable_names') or []
        if not label.get('w_bit'): label = dict(label, w_bit=cc.get('w_bit'), a_bit=cc.get('a_bit'))
        primary_w = label.get('w_bit')
    rows=[]
    for mt in get_model_types(cfg):
        s = mspec(mt, found)
        r = dict(yaml=yaml_path.split('lab_practice/')[1], model=mt,
                 w=label.get('w_bit'), a=label.get('a_bit'), kvc=kvc_label)
        if s is None or s['L']==0:
            for k in ('attn','moe','mlp','shared'):
                r[f'{k}_fp_layers']=None; r[f'{k}_fp_params']='N/A'; r[f'{k}_up']=''
                r[f'{k}_tot']=None; r[f'{k}_dtypes']='N/A'; r[f'{k}_up_list']=[]
            for k in ('attn','moe','mlp','shared'): r[f'{k}_act']='N/A'
            r.update(fa3_fp_layers=None, fa3_fp_detail='N/A', fa3_up='',
                     kvc_rb_layers=None, kvc_rb_detail='N/A',
                     lin_total=None, fp_par_all=None, up_par_all=None, eq_w=None, eq_a=None,
                     act_fp_par_all=None, act_up_par_all=None,
                     L=0, attn_layers_total=0, moe_layers_total=0, dense_layers_total=0, mtp_layers=0,
                     fa_dtypes='N/A', fa_up_list=[], kvc_dtype=None,
                     plan_dtypes='N/A', fa_plan_dtypes='N/A', fa_bit=primary_fa,
                     la_fp_layers=None, la_fp_params=None, la_act_fp_layers=None, la_act_up_list=[], la_up_list=[], la_partial_layers=0,
                     note=('通用模板配置（适用任意HF transformers模型，无特定目标模型）；include:* 覆盖全部线性层、不存在回退，但无具体模型无法核算参数量' if mt=='transformers' else '无公开config，不适用'))
            rows.append(r); continue
        L_eff = s['L'] + s['mtp']
        inv={}
        for i in range(L_eff):
            ii = min(i, s['L']-1)
            for key,(p,cat,aliases) in layer_modules(s, ii).items():
                inv[(i,key)]=(p,cat,[f"model.layers.{i}.{a}" for a in aliases] if not s.get('dit') else [f"blocks.{i}.{a}" for a in aliases])
        if s.get('has_vision'):
            inv[(-1,'vision')]=(None,'vision',['vision.*','visual.*','vision_tower.*'])
        # 计划方案：配置中位宽不超过 label 主位宽的 dtype（先生效者优先，后面的块可能实际未生效）
        lin_plan_dtypes = sorted({(wd or 'int8') for (t,wd,inc,exc,fd,ad) in lin_blocks
                                  if wd and BITS.get(wd,99) <= (primary_w or 16)})
        fa_plan_dtypes = sorted({fd for (t,wd,inc,exc,fd,ad) in fa3_blocks
                                 if fd and BITS.get(fd,99) <= (primary_fa or 16)})
        # 失效块检测：按顺序模拟"先生效者优先"，统计每个块新覆盖的模块数
        dead_blocks=[]
        if lin_blocks and not spec_cfg.get('per_expert'):  # per_expert 是双模型（高低噪）分别的配置，不做失效判定
            claimed=set()
            for (t,wd,inc,exc,fd,ad) in lin_blocks:
                if any(any(k in p for k in ('vision','visual','embed','lm_head','mtp')) for p in inc):
                    continue  # 我的模块清单不含 vision/embed/mtp.* 的真实命名，不做失效判定
                new=0; n_raw=0
                for (li,key),(p,cat,names) in inv.items():
                    if cat=='router': continue
                    if any(match_any(inc,n) for n in names) and not any(match_any(exc,n) for n in names):
                        n_raw+=1
                        if (li,key) not in claimed:
                            claimed.add((li,key)); new+=1
                if new==0 and n_raw>0: dead_blocks.append(f"{wd or 'int8'}(已被先前块覆盖)")
                elif new==0: dead_blocks.append(f"{wd or 'int8'}(模型中无匹配模块)")
        def status(names):
            # 返回 (weight_dtype 或 'FP', act_dtype 或 None 或 'NA')
            # act None = 权重量化了但激活未量化; 'NA' = 模块未覆盖(整体FP)
            if legacy_dis is not None:
                if any(match_any(legacy_dis, n) for n in names): return 'FP','NA'
                return f"int{label.get('w_bit') or 8}", (f"int{label.get('a_bit')}" if label.get('a_bit') and label.get('a_bit')<16 else None)
            for (t,wd,inc,exc,fd,ad) in lin_blocks:
                included = any(match_any(inc,n) for n in names)
                excluded = any(match_any(exc,n) for n in names)
                if included and not excluded: return wd or 'int8', ad
            return 'FP','NA'
        CATS=('attn','moe','dense_mlp','shared')
        fp={c:set() for c in CATS}; fp_par={c:0 for c in CATS}; fp_unk={c:False for c in CATS}
        up={c:{} for c in CATS}
        cat_tot={c:0 for c in CATS}; cat_unk={c:False for c in CATS}
        cat_dtypes={c:set() for c in CATS}
        act_fp={c:set() for c in CATS}; act_up={c:{} for c in CATS}
        other_fp={'shared':set(),'vision':set(),'linear_attn':set(),'mod':set()}
        other_cov={'shared':set(),'vision':set(),'linear_attn':set(),'mod':set()}
        other_fp_par={'shared':0,'vision':0,'linear_attn':0,'mod':0}
        other_act_fp={'shared':set(),'vision':set(),'linear_attn':set(),'mod':set()}
        other_act_up={}
        other_up={}
        label_a = label.get('a_bit')
        tot_par=0; eq_num=0; eqa_num=0; fp_par_all=0; up_par_all=0; unk=False
        act_fp_par_all=0; act_up_par_all=0
        for (li,key),(p,cat,names) in inv.items():
            if cat=='router': continue
            st, ad = status(names)
            bits = 16 if st=='FP' else BITS.get(st, 8)
            is_up = st!='FP' and primary_w and bits > primary_w
            # 激活侧：模块未被覆盖(整体FP，权重激活一起回退)或被覆盖但 act 未量化 = 激活浮点回退;
            # act 位宽高于 label a_bit = 激活升位宽回退
            a_bits = 16 if st=='FP' else (16 if ad is None else BITS.get(ad,8))
            is_act_fp = (st=='FP' or ad is None) and (label_a or 16) < 16
            is_act_up = st!='FP' and ad is not None and label_a and a_bits > label_a
            if p:
                tot_par += p; eq_num += bits*p; eqa_num += a_bits*p
                if st=='FP': fp_par_all += p
                elif is_up: up_par_all += p
                if is_act_fp: act_fp_par_all += p
                elif is_act_up: act_up_par_all += p
            else:
                unk = True
            if cat in CATS:
                if p: cat_tot[cat]+=p
                else: cat_unk[cat]=True
                if st=='FP':
                    fp[cat].add(li)
                    if p: fp_par[cat]+=p
                    else: fp_unk[cat]=True
                elif is_up:
                    dd=up[cat].setdefault(st,[set(),0]); dd[0].add(li); dd[1]+=p or 0
                if st!='FP': cat_dtypes[cat].add(st)
                if is_act_fp: act_fp[cat].add(li)
                elif is_act_up:
                    dd=act_up[cat].setdefault(ad,[set(),0]); dd[0].add(li); dd[1]+=p or 0
            else:
                if st=='FP':
                    other_fp[cat].add(li); other_fp_par[cat]+=p or 0
                    if is_act_fp: other_act_fp[cat].add(li)
                elif is_up:
                    dd=other_up.setdefault((cat, st),[set(),0]); dd[0].add(li); dd[1]+=p or 0
                    if is_act_up: other_act_up.setdefault((cat, ad),set()).add(li)
                else:
                    other_cov[cat].add(li)
                    if is_act_fp: other_act_fp[cat].add(li)
                    elif is_act_up: other_act_up.setdefault((cat, ad),set()).add(li)
        for c,short in (('attn','attn'),('moe','moe'),('dense_mlp','mlp'),('shared','shared')):
            r[f'{short}_fp_layers']=len(fp[c])
            r[f'{short}_fp_params']=fp_par[c] if not fp_unk[c] else ('%d+'%fp_par[c] if fp_par[c] else 'N/A')
            r[f'{short}_up']='; '.join(f"{len(v[0])}层/{(v[1]/1e9):.2f}B→{k}" for k,v in sorted(up[c].items()))
            r[f'{short}_tot']=None if (cat_unk[c] and cat_tot[c]==0) else cat_tot[c]
            r[f'{short}_dtypes']='+'.join(sorted(cat_dtypes[c])) if cat_dtypes[c] else ('FP（未量化）' if fp[c] else '')
            r[f'{short}_up_list']=[(k, len(v[0]), v[1]) for k,v in sorted(up[c].items())]
            act_parts=[]
            if act_fp[c]: act_parts.append(f"FP:{len(act_fp[c])}层")
            for k,v in sorted(act_up[c].items()):
                act_parts.append(f"升位宽:{len(v[0])}层/{(v[1]/1e9):.2f}B→{k}")
            r[f'{short}_act']='; '.join(act_parts)
        r['L']=s['L']; r['attn_layers_total']=len(s['attn_layers']); r['moe_layers_total']=len(s['moe_layers'])
        r['dense_layers_total']=len(s['dense_layers']); r['mtp_layers']=s['mtp']
        # 性能损失指标
        r['lin_total']=tot_par
        r['fp_par_all']=fp_par_all
        r['up_par_all']=up_par_all
        r['act_fp_par_all']=act_fp_par_all
        r['act_up_par_all']=act_up_par_all
        r['eq_w']=(eq_num/tot_par) if tot_par else None
        r['eq_a']=(eqa_num/tot_par) if tot_par else None
        r['unk']=unk
        rem=[]
        if other_fp['shared']: rem.append(f"shared_experts浮点回退{len(other_fp['shared'])}层")
        if other_fp['vision']: rem.append("vision未量化")
        # 线性注意力回退并入 attn 列统计，此处只保留结构化数据
        r['la_fp_layers']=len(other_fp['linear_attn'])
        r['la_fp_params']=other_fp_par['linear_attn']
        r['la_act_fp_layers']=len(other_act_fp['linear_attn'])
        r['la_act_up_list']=[(dt,len(v)) for (c,dt),v in sorted(other_act_up.items()) if c=='linear_attn']
        r['la_up_list']=[(dt,len(v[0]),v[1]) for (c,dt),v in sorted(other_up.items()) if c=='linear_attn']
        r['la_partial_layers']=len(other_fp['linear_attn'] & other_cov['linear_attn'])
        if other_fp['mod']: rem.append(f"调制层(mod)浮点回退{len(other_fp['mod'])}块")
        if other_up:
            CAT_ZH={'shared':'共享专家','vision':'视觉模块','mod':'调制层'}
            ou = {k:v for k,v in other_up.items() if k[0]!='linear_attn'}
            if ou: rem.append("其他升位宽:"+"，".join(f"{CAT_ZH.get(c,c)}{len(v[0])}块→{dt}" for (c,dt),v in sorted(ou.items())))
        low_dt = sorted({dt for c in CATS for dt in cat_dtypes[c] if primary_w and BITS.get(dt,99)<primary_w})
        if low_dt: rem.append(f"存在低于label位宽的量化:{'+'.join(low_dt)}（超额压缩，非回退，等效位宽可低于label）")
        if legacy_dis is not None: rem.append("V0旧格式(disable_names)")
        if not lin_blocks and legacy_dis is None:
            btypes = {b[0] for b in blocks}
            if 'float_sparse' in btypes:
                mratio = re.search(r'sparse_ratio:\s*([\d.]+)', raw)
                pct = f"{float(mratio.group(1))*100:.0f}%" if mratio else ''
                rem.append(f"无位宽量化：仅做浮点稀疏（{pct}权重置零），目标即FP16，不在位宽量化回退口径内，FP不计为回退")
            elif not blocks:
                rem.append("无量化块，整体保持浮点")
        if dead_blocks: rem.append(f"量化块实际未生效:{'+'.join(sorted(set(dead_blocks)))}")
        if s.get('dit') and mt.startswith('Wan2.2') and 'A14B' in mt: rem.append("高低噪双专家，块数按单个DiT计")
        r['plan_dtypes']='+'.join(lin_plan_dtypes)
        r['fa_plan_dtypes']='+'.join(fa_plan_dtypes)
        r['fa_bit']=primary_fa
        if fa3_blocks:
            fa_fp=set(); fa_up={}
            fa_dtypes=set()
            for li in range(L_eff):
                nm = f"blocks.{li}.self_attn" if s.get('dit') else f"model.layers.{li}.self_attn"
                st='FP'
                for (t,wd,inc,exc,fd,ad) in fa3_blocks:
                    if match_any(inc,nm) and not match_any(exc,nm): st=fd or 'int8'; break
                if st=='FP': fa_fp.add(li)
                else:
                    fa_dtypes.add(st)
                    if primary_fa and BITS.get(st,0)>primary_fa: fa_up.setdefault(st,set()).add(li)
            r['fa3_fp_layers']=len(fa_fp); r['fa3_fp_detail']=sorted(fa_fp)
            r['fa3_up']='; '.join(f"{len(v)}层→{k}" for k,v in fa_up.items())
            r['fa_dtypes']='+'.join(sorted(fa_dtypes)) if fa_dtypes else 'FP（未量化）'
            r['fa_up_list']=[(k,len(v)) for k,v in sorted(fa_up.items())]
        else:
            r['fa3_fp_layers']=None; r['fa3_fp_detail']='未启用fa3量化'; r['fa3_up']=''
            r['fa_dtypes']=None; r['fa_up_list']=[]
        if not kvc_label:
            r['kvc_rb_layers']=None; r['kvc_rb_detail']='无KVcache量化'; r['kvc_dtype']=None
        else:
            kset=set(); kvc_dtype=None
            for (t,wd,inc,exc,fd,ad) in cache_blocks:
                if fd and not kvc_dtype: kvc_dtype=fd
                excL=layers_in_patterns(exc)
                if excL: kset|=excL
            if not cache_blocks and fa3_blocks and ('knope' in raw or 'kv_cache compression' in raw.lower()):
                for (t,wd,inc,exc,fd,ad) in fa3_blocks:
                    if fd and not kvc_dtype: kvc_dtype=fd
                    excL=layers_in_patterns(exc)
                    if excL: kset|=excL
            r['kvc_rb_layers']=len(kset); r['kvc_rb_detail']=sorted(kset)
            r['kvc_dtype']=kvc_dtype or 'int8'
        r['note']=';'.join(rem)
        rows.append(r)
    return rows
