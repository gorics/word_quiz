from __future__ import annotations
import gc, hashlib, json, os, time
from pathlib import Path
import torch

OUT=Path(os.environ.get('OUT_DIR','flybrain-hybrid/results'))
OUT.mkdir(parents=True,exist_ok=True)
report=json.load(open(OUT/'brain_report.json','r',encoding='utf-8'))
llm_prompt=report['llm_prompt']
sd_prompt=report['stable_diffusion_prompt']
summary={'brain_controller':report['controller'],'top_groups':report['top_groups'][:6],'models':{},'timings':{}}
torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))

t0=time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM
llm_id='HuggingFaceTB/SmolLM2-360M-Instruct'
tok=AutoTokenizer.from_pretrained(llm_id)
model=AutoModelForCausalLM.from_pretrained(llm_id,torch_dtype=torch.float32,low_cpu_mem_usage=True)
messages=[{'role':'user','content':llm_prompt}]
text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
inputs=tok(text,return_tensors='pt')
with torch.inference_mode():
    out=model.generate(**inputs,max_new_tokens=64,do_sample=False,repetition_penalty=1.08,pad_token_id=tok.eos_token_id)
answer=tok.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
(OUT/'llm_output.txt').write_text(answer+'\n',encoding='utf-8')
summary['models']['llm']={'id':llm_id,'output':answer,'prompt':llm_prompt}
summary['timings']['llm_seconds']=time.time()-t0

del model,tok,inputs,out
gc.collect()

t0=time.time()
from diffusers import StableDiffusionPipeline
sd_id='segmind/tiny-sd'
pipe=StableDiffusionPipeline.from_pretrained(sd_id,torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe=pipe.to('cpu')
pipe.set_progress_bar_config(disable=False)
seed=int(abs(sum(report['controller'].values()))*1_000_000)%2_147_483_647
gen=torch.Generator(device='cpu').manual_seed(seed)
with torch.inference_mode():
    image=pipe(sd_prompt,height=256,width=256,num_inference_steps=6,guidance_scale=6.0,generator=gen).images[0]
image_path=OUT/'stable_diffusion_output.png'
image.save(image_path)
summary['models']['stable_diffusion']={'id':sd_id,'prompt':sd_prompt,'seed':seed,'size':[256,256],'steps':6,'guidance_scale':6.0}
summary['timings']['stable_diffusion_seconds']=time.time()-t0
summary['sha256']={'image':hashlib.sha256(image_path.read_bytes()).hexdigest(),'llm_output':hashlib.sha256((OUT/'llm_output.txt').read_bytes()).hexdigest()}
json.dump(summary,open(OUT/'generation_report.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))
