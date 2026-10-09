# -*- coding: utf-8 -*-
"""从 ModelScope 抓取模型 config.json，带本地缓存"""
import os, sys, json, glob
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import engine

WORK = os.environ.get('MSLIM_WORK', os.path.join(os.getcwd(), 'work'))
REPO = os.environ.get('MSLIM_REPO', os.path.join(WORK, 'msmodelslim'))
CACHE = os.path.join(WORK, 'configs')
os.makedirs(CACHE, exist_ok=True)

ALIAS = {
 'GLM-5.3':'ZhipuAI/GLM-5.3',
 'GLM-5.3-Flash':'ZhipuAI/GLM-5.3-Flash',
 'Qwen3.6-35B-A3B':'Qwen/Qwen3.6-35B-A3B',
 'DeepSeek-V4-Flash-0731':'deepseek-ai/DeepSeek-V4-Flash',
 'Qwen3.5-27B':'Qwen/Qwen3.5-27B',
 'Qwen3-235B':'Qwen/Qwen3-235B-A22B',
 'Qwen3-30B':'Qwen/Qwen3-30B-A3B',
 'Qwen3-Coder-480B-A35B':'Qwen/Qwen3-Coder-480B-A35B-Instruct',
 'Qwen3-VL-235B-A22B':'Qwen/Qwen3-VL-235B-A22B-Instruct',
 'Qwen3-VL-30B-A3B':'Qwen/Qwen3-VL-30B-A3B-Instruct',
 'Qwen2.5-Omni-7B':'Qwen/Qwen2.5-Omni-7B',
 'Qwen3-Omni-30B-A3B-Instruct':'Qwen/Qwen3-Omni-30B-A3B-Instruct',
 'Qwen3-Omni-30B-A3B-Thinking':'Qwen/Qwen3-Omni-30B-A3B-Instruct',
 'InternVL3_5-241B-A28B':'OpenGVLab/InternVL3_5-241B-A28B',
 'InternVL3_5-38B':'OpenGVLab/InternVL3_5-38B',
}

def candidates(name):
    cands=[]
    if name in ALIAS: cands.append(ALIAS[name])
    orgs=[]
    n=name
    low=n.lower()
    if low.startswith('qwen') or 'qwq' in low: orgs+=['Qwen','LLM-Research','AI-ModelScope']
    if low.startswith('deepseek'): orgs+=['deepseek-ai','AI-ModelScope']
    if low.startswith('glm') or 'chatglm' in low: orgs+=['ZhipuAI','AI-ModelScope']
    if low.startswith('kimi') or 'moonshot' in low: orgs+=['moonshotai','AI-ModelScope']
    if low.startswith('minimax'): orgs+=['MiniMaxAI','MiniMax']
    if low.startswith('step'): orgs+=['stepfun-ai']
    if low.startswith('intern'): orgs+=['OpenGVLab','internlm']
    if low.startswith('gemma'): orgs+=['LLM-Research','AI-ModelScope','google']
    if low.startswith('hy') or 'hunyuan' in low: orgs+=['Tencent-Hunyuan','hunyuan']
    if 'longcat' in low: orgs+=['meituan-longcat','LongCat']
    if 'mimo' in low: orgs+=['XiaomiMiMo','MiMo']
    orgs+=['AI-ModelScope','LLM-Research','iic','modelscope']
    seen=set(); orgs=[o for o in orgs if not (o in seen or seen.add(o))]
    names=[n]
    for pre in ('Qwen-','DeepSeek-','GLM-','Kimi-','MiniMax-','Step-','InternVL-','Hy-'):
        if n.startswith(pre): names.append(n[len(pre):])
    for nm in names:
        for o in orgs: cands.append(f'{o}/{nm}')
        cands.append(nm)
    return cands

def fetch_one(name):
    dst=os.path.join(CACHE, name.replace('/','__')+'.json')
    if os.path.exists(dst):
        try:
            json.load(open(dst)); return name, dst
        except Exception: os.remove(dst)   # 损坏缓存则重抓
    for path in candidates(name):
        url=f'https://www.modelscope.cn/models/{path}/resolve/master/config.json'
        try:
            req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as r:
                txt=r.read().decode('utf-8')
            j=json.loads(txt)
            def extract(j):
                has_vis = bool(j.get('vision_config') or j.get('visual'))
                cur=j
                for _ in range(3):
                    if cur.get('hidden_size') and cur.get('num_hidden_layers'):
                        cur=dict(cur); cur['_has_vision']=has_vis or bool(cur.get('vision_config'))
                        return cur
                    nxt=None
                    for k in ('llm_config','text_config','thinker_config','language_config'):
                        if isinstance(cur.get(k), dict): nxt=cur[k]; break
                    if nxt is None: return None
                    has_vis = has_vis or bool(cur.get('vision_config') or cur.get('visual'))
                    cur=nxt
                return None
            j2=extract(j)
            if j2:
                tmp=dst+'.tmp'
                with open(tmp,'w') as f: json.dump(j2, f)   # 显式关闭
                os.replace(tmp, dst)                        # 原子落盘
                json.load(open(dst))                        # 落盘校验
                return name, dst
        except Exception: pass
    return name, None

if __name__=='__main__':
    data = engine.load_yamls(os.path.join(REPO, 'lab_practice'))
    names=set()
    for p,cfg in data.items():
        for mt in engine.get_model_types(cfg): names.add(mt)
    names=sorted(names)
    print('models:', len(names))
    found={}; missing=[]
    with ThreadPoolExecutor(16) as ex:
        for name, dst in ex.map(fetch_one, names):
            if dst:
                try: found[name]=json.load(open(dst)); continue
                except Exception: pass
            missing.append(name)
    # 并发失败/幻影的串行重试（网络挂载元数据延迟 + 限频）
    import time
    for rnd in range(4):
        if not missing: break
        retry, missing = missing, []
        for name in retry:
            dst=os.path.join(CACHE, name.replace('/','__')+'.json')
            if os.path.exists(dst):
                try: found[name]=json.load(open(dst)); continue
                except Exception: os.remove(dst)
            name_, dst = fetch_one(name)
            if dst:
                try: found[name]=json.load(open(dst)); continue
                except Exception: pass
            missing.append(name)
        if missing: time.sleep(4)
    with open(os.path.join(WORK, 'found_index.json'), 'w') as f:
        json.dump({k:v for k,v in found.items()}, f)
    print('fetched:', len(found))
    print('missing:', missing)
