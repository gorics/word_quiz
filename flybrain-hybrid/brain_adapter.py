from __future__ import annotations
import argparse, gzip, json, struct, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

SCENARIOS = {
    "visual_explore": ({"VIS_R1R6": 1.0, "VIS_ME": 0.8, "DRIVE_HUNGER": 0.15}, [0.25,0.55,0.45,0.50,0.85,0.70]),
    "food_seek": ({"OLF_ORN_FOOD": 1.0, "OLF_PN": 0.5, "DRIVE_HUNGER": 0.9}, [0.85,0.65,0.55,0.55,0.65,0.55]),
    "mechanical_touch": ({"MECH_BRISTLE": 1.0, "MECH_JO": 0.6}, [-0.35,0.95,0.90,0.45,0.55,0.35]),
    "warmth": ({"THERMO_WARM": 1.0}, [0.15,0.45,0.25,0.95,0.60,0.40]),
    "cool": ({"THERMO_COOL": 1.0}, [0.05,0.35,0.20,0.05,0.65,0.45]),
    "central_memory": ({"MB_KC": 1.0, "MB_DAN_REW": 0.35}, [0.45,0.35,0.15,0.50,0.95,0.80]),
    "navigation": ({"CX_EPG": 1.0, "CX_PFN": 0.8, "CX_FC": 0.6}, [0.25,0.60,0.75,0.50,0.80,0.75]),
    "feeding_motor": ({"GUS_GRN_SWEET": 1.0, "SEZ_FEED": 0.7, "MN_PROBOSCIS": 0.5}, [0.75,0.70,0.50,0.60,0.55,0.35]),
}
CONTROL_NAMES = ["valence","arousal","motion","warmth","detail","curiosity"]

def load_connectome(bin_gz: Path, meta_json: Path):
    t0=time.time(); raw=gzip.open(bin_gz,'rb').read(); n,e=struct.unpack_from('<II',raw,0)
    edge_dtype=np.dtype([('pre','<u4'),('post','<u4'),('w','<f4')]); edges=np.frombuffer(raw,dtype=edge_dtype,count=e,offset=8)
    meta_off=8+e*12; meta_dtype=np.dtype([('region','u1'),('group','<u2')]); neuron_meta=np.frombuffer(raw,dtype=meta_dtype,count=n,offset=meta_off)
    meta=json.load(open(meta_json,'r',encoding='utf-8')); G=meta['group_count']
    pre_g=neuron_meta['group'][edges['pre']].astype(np.int64); post_g=neuron_meta['group'][edges['post']].astype(np.int64); flat=pre_g*G+post_g
    W=np.bincount(flat,weights=edges['w'].astype(np.float64),minlength=G*G).reshape(G,G)
    A=np.bincount(flat,weights=np.abs(edges['w']).astype(np.float64),minlength=G*G).reshape(G,G)
    Wn=W/np.maximum(A.sum(1,keepdims=True),1.0); names=[g['name'] for g in meta['groups']]
    return n,e,meta,names,W,Wn,time.time()-t0

def simulate(names,Wn,stim,steps=32,leak=0.82,gain=2.25):
    idx={n:i for i,n in enumerate(names)}; u=np.zeros(len(names),dtype=np.float64)
    for name,val in stim.items():
        if name in idx: u[idx[name]]=val
    x=np.zeros_like(u); traj=[]
    for t in range(steps):
        drive=u*(1.0 if t < max(5,steps//3) else 0.18); recurrent=Wn.T@x
        x=leak*x+(1-leak)*np.tanh(gain*(recurrent+drive)); x=np.where(np.abs(x)<0.015,0,x); traj.append(x.copy())
    return np.stack(traj),x

class Adapter(nn.Module):
    def __init__(self,g):
        super().__init__(); self.net=nn.Sequential(nn.Linear(g,48),nn.GELU(),nn.Linear(48,24),nn.GELU(),nn.Linear(24,6),nn.Tanh())
    def forward(self,x): return self.net(x)

def train_adapter(states,targets,seed=7):
    torch.manual_seed(seed); np.random.seed(seed); X=[];Y=[]
    for s,t in zip(states,targets):
        s=s/max(float(np.max(np.abs(s))),1e-6)
        for _ in range(96): X.append((s+np.random.normal(0,0.025,size=s.shape)).astype('float32'));Y.append(np.asarray(t,dtype='float32'))
    X=torch.tensor(np.stack(X));Y=torch.tensor(np.stack(Y));m=Adapter(X.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4);hist=[]
    for ep in range(260):
        pred=m(X);loss=((pred-Y)**2).mean();opt.zero_grad();loss.backward();opt.step()
        if ep%20==0 or ep==259: hist.append([ep,float(loss.detach())])
    return m,hist,float(((m(X)-Y)**2).mean().detach())

def describe(names,state,meta,k=8):
    top=np.argsort(np.abs(state))[-k:][::-1]; groups=[{"group":names[i],"activation":float(state[i])} for i in top]
    regions={r:0.0 for r in ['sensory','central','drives','motor']};counts={r:0 for r in regions}
    for i,g in enumerate(meta['groups']):
        if g['neuron_count']>0: regions[g['region']]+=float(abs(state[i]));counts[g['region']]+=1
    for r in regions: regions[r]/=max(counts[r],1)
    return groups,regions

def controls_to_prompts(label,controls,top_groups):
    d=dict(zip(CONTROL_NAMES,[float(x) for x in controls])); mood='positive' if d['valence']>0.3 else ('aversive' if d['valence']<-0.2 else 'neutral')
    temp='warm amber' if d['warmth']>0.65 else ('cool cyan' if d['warmth']<0.3 else 'balanced natural');motion='dynamic motion' if d['motion']>0.65 else 'still composition';detail='highly detailed macro' if d['detail']>0.7 else 'simple macro';brain=', '.join(g['group'] for g in top_groups[:4])
    llm=(f"You are a compact language model coupled to a simulated Drosophila connectome. Current state={label}; dominant circuits={brain}; valence={d['valence']:.2f}, arousal={d['arousal']:.2f}, motion={d['motion']:.2f}, curiosity={d['curiosity']:.2f}. In one vivid sentence, describe the agent's current perception and intended action. Do not claim consciousness; describe it as a simulation.")
    sd=(f"{detail} photograph from a fruit fly inspired compound-eye perspective, {mood} behavioral state, {temp} lighting, {motion}, faceted vision, microscopic environmental textures, scientifically inspired neural aesthetic, dominant neural motifs {brain}, cinematic depth of field")
    return llm,sd,d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--scenario',default='food_seek');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    n,e,meta,names,W,Wn,load_s=load_connectome(args.data/'connectome.bin.gz',args.data/'neuron_meta.json');states=[];targets=[];scenario_reports={}
    for name,(stim,target) in SCENARIOS.items():
        _,state=simulate(names,Wn,stim);states.append(state);targets.append(target);top,regions=describe(names,state,meta);scenario_reports[name]={"stimulus":stim,"top_groups":top,"region_activity":regions,"final_l2":float(np.linalg.norm(state))}
    model,hist,mse=train_adapter(states,targets);torch.save({"state_dict":model.state_dict(),"group_names":names,"controls":CONTROL_NAMES},args.out/'flybrain_adapter.pt')
    chosen=args.scenario if args.scenario in SCENARIOS else 'food_seek';state=states[list(SCENARIOS).index(chosen)];norm=state/max(float(np.max(np.abs(state))),1e-6)
    with torch.no_grad(): ctrl=model(torch.tensor(norm,dtype=torch.float32)).numpy()
    top,regions=describe(names,state,meta);llm_prompt,sd_prompt,ctrl_dict=controls_to_prompts(chosen,ctrl,top)
    report={"architecture":"full FlyWire edge aggregation -> signed 63-group recurrent dynamics -> trained MLP control adapter -> LLM/Stable-Diffusion prompts","connectome":{"neurons":n,"edges":e,"groups":meta['group_count'],"load_seconds":load_s,"nonzero_group_links":int(np.count_nonzero(W))},"adapter_training":{"examples":len(SCENARIOS)*96,"epochs":260,"mse":mse,"loss_curve":hist,"controls":CONTROL_NAMES},"selected_scenario":chosen,"controller":ctrl_dict,"top_groups":top,"region_activity":regions,"llm_prompt":llm_prompt,"stable_diffusion_prompt":sd_prompt,"scenarios":scenario_reports}
    json.dump(report,open(args.out/'brain_report.json','w',encoding='utf-8'),indent=2,ensure_ascii=False);np.save(args.out/'group_connectivity.npy',W.astype('float32'));print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
