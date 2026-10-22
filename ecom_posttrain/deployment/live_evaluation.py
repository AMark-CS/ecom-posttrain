"""Measured single-request CUDA benchmark and optional lm-eval task evaluation."""
import hashlib
from collections import Counter
import importlib.metadata
import importlib.util
import math
import json
import sys
from pathlib import Path
import platform
import statistics
import time

SUFFIXES={'none':'','awq':'-AWQ','gptq-int4':'-GPTQ-Int4','gptq-int8':'-GPTQ-Int8'}


def task_metrics(report, task):
    values=report['results'][task]
    def number(key):
        value=values.get(key)
        return float(value) if isinstance(value,(float,int)) and math.isfinite(value) else None
    return dict(task=task,acc=number('acc,none'),acc_norm=number('acc_norm,none'),
                acc_stderr=number('acc_stderr,none'),acc_norm_stderr=number('acc_norm_stderr,none'),
                samples=report.get('n-samples',{}).get(task,{}),version=report.get('versions',{}).get(task))


def run(cfg, adapter, emit, cancelled, restore_adapter):
    import torch
    import transformers
    from huggingface_hub import snapshot_download,hf_hub_download
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from nano_posttrain.utils.seed import set_seed

    if not torch.cuda.is_available():
        raise ValueError('Deployment measurements require CUDA; CPU results are not substituted')
    requirements=[]
    if cfg['mode'] != 'benchmark': requirements.append('lm_eval')
    if cfg['attention']=='flash_attention_2': requirements.append('flash_attn')
    if cfg['quantization'].startswith('gptq'): requirements.append('gptqmodel')
    if cfg['quantization']=='awq': requirements.append('gptqmodel' if int(transformers.__version__.split('.')[0])>=5 else 'awq')
    missing=[name for name in requirements if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError('Missing deployment dependencies: '+', '.join(missing)+'. Use a compatible isolated environment; no fallback was applied.')
    set_seed(cfg['seed'])
    model_id=cfg['model']+SUFFIXES[cfg['quantization']]
    revision=adapter.get('revision') if adapter else None
    if 'Qwen2.5-7B' in model_id and cfg['quantization']=='none':
        index=Path(hf_hub_download(model_id,'model.safetensors.index.json',revision=revision,local_files_only=not cfg['allow_download']))
        weight_bytes=json.loads(index.read_text())['metadata']['total_size']
        total_bytes=torch.cuda.get_device_properties(0).total_memory
        if weight_bytes>total_bytes:
            raise ValueError(f'Not loaded: checkpoint weights alone require {weight_bytes/1024**3:.2f} GiB, exceeding this GPU\'s {total_bytes/1024**3:.2f} GiB. No CPU offload or lower precision was substituted.')
        revision=index.parent.name
    emit('phase',message=f'Resolving {model_id}; {cfg["attention"]}, {cfg["dtype"]}')
    path=snapshot_download(model_id,revision=revision,
                           local_files_only=not cfg['allow_download'],
                           allow_patterns=['*.json','*.txt','*.safetensors','*.jinja'])
    if cancelled(): return
    revision=path.replace('\\','/').split('/')[-1]
    torch.cuda.reset_peak_memory_stats()
    free_before_load=torch.cuda.mem_get_info()[0]/1024**2
    tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=False)
    dtype=getattr(torch,cfg['dtype'])
    emit('phase',message='Loading deployment model on one GPU; CPU offload is disabled')
    model=AutoModelForCausalLM.from_pretrained(path,dtype=dtype,device_map='cuda:0',
                                             attn_implementation=cfg['attention'],trust_remote_code=False)
    quant=getattr(model.config,'quantization_config',{})
    if hasattr(quant,'to_dict'): quant=quant.to_dict()
    if cfg['quantization']!='none':
        expected='awq' if cfg['quantization']=='awq' else 'gptq'
        bits=8 if cfg['quantization']=='gptq-int8' else 4
        if quant.get('quant_method')!=expected or quant.get('bits')!=bits:
            raise ValueError('Loaded checkpoint quantization metadata differs from the requested configuration')
    if adapter:
        from nano_posttrain.optim import apply_lora
        apply_lora(model,r=adapter['lora_r'],alpha=adapter['lora_alpha'],dropout=0)
        model.to('cuda:0')
        restore_adapter(model,adapter['path'])
    model.eval();model.config.use_cache=True
    actual_attention=getattr(model.config,'_attn_implementation',None)
    if actual_attention!=cfg['attention']:
        raise ValueError(f'Attention implementation changed to {actual_attention}; refusing a mislabeled benchmark')
    if model.dtype != dtype:
        raise ValueError(f'Loaded compute dtype {model.dtype} differs from requested {dtype}; select its actual supported precision')
    env=dict(gpu=torch.cuda.get_device_name(0),vram_mb=round(torch.cuda.get_device_properties(0).total_memory/1024**2),
             torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
             python=platform.python_version(),system=platform.system(),
             gpu_capability=list(torch.cuda.get_device_capability(0)))
    for package in ['flash_attn','gptqmodel','autoawq','triton','optimum']:
        try: env[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    kernels=dict(Counter(type(m).__module__+'.'+type(m).__name__ for m in model.modules() if 'quantlinear' in type(m).__name__.lower() or 'wqlinear' in type(m).__name__.lower()))
    awq_module=sys.modules.get('awq.modules.linear.gemm')
    if cfg['quantization']=='awq' and awq_module:
        kernels['awq_backend']='CUDA extension' if awq_module.awq_ext is not None else 'Triton' if awq_module.TRITON_AVAILABLE else 'naive'
        if kernels['awq_backend']=='naive':
            raise ValueError('No accelerated AWQ kernel is available; refusing a silently dequantized benchmark')
    load_peak=torch.cuda.max_memory_allocated()/1024**2
    protocol=dict(batch_size=1,input_tokens=cfg['input_tokens'],decode_tokens=50,warmup=cfg['warmup'],
                  repetitions=cfg['repetitions'],seed=cfg['seed'],num_fewshot=0,apply_chat_template=False,
                  limit=cfg['limit'],evaluation_max_length=2048,
                  decode_definition='50 greedy output tokens: first from prefill logits, followed by 49 KV-cached forward passes; EOS ignored',
                  memory_definition='CUDA peak allocated/reserved includes resident weights; phases reset separately; no CPU offload')
    protocol['memory_residency']='CUDA allocation is not physical residency. WSL/WDDM can page allocations into system memory; allocations above physical GPU capacity are rejected for task evaluation.'
    base_ids=tokenizer('Describe a stainless steel travel bottle priced at $24.99. Use only supplied product facts. ',add_special_tokens=False)['input_ids']
    ids=(base_ids*(cfg['input_tokens']//len(base_ids)+1))[:cfg['input_tokens']]
    protocol['prompt_sha256']=hashlib.sha256(bytes(str(ids),'utf-8')).hexdigest()
    emit('deployment_info',model=model_id,revision=revision,environment=env,protocol=protocol,
         attention=actual_attention,quantization=quant,adapter=cfg['adapter'],kernels=kernels,
         free_vram_before_load_mb=free_before_load,compute_dtype=str(model.dtype))
    inputs=torch.tensor([ids],device='cuda')
    rows=[];scores=[]

    def iteration(step,record):
        if cancelled(): return
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.inference_mode():
            output=model(inputs,use_cache=True,logits_to_keep=1)
            token=output.logits[:,-1].argmax(-1,keepdim=True)
            cache=output.past_key_values
            del output
            torch.cuda.synchronize()
            prefill=time.perf_counter()-start
            prefill_peak=torch.cuda.max_memory_allocated()/1024**2
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            # The first generated token comes from prefill; generate the remaining 49 with the KV cache.
            for _ in range(49):
                if cancelled(): return
                output=model(token,past_key_values=cache,use_cache=True,logits_to_keep=1)
                token=output.logits[:,-1].argmax(-1,keepdim=True)
                cache=output.past_key_values
                del output
            torch.cuda.synchronize();decode=time.perf_counter()-start
        row=dict(step=step,prefill_ms=prefill*1000,decode_ms=decode*1000,
                 prefill_tps=cfg['input_tokens']/prefill,decode_tps=50/decode,
                 prefill_peak_mb=prefill_peak,decode_peak_mb=torch.cuda.max_memory_allocated()/1024**2,
                 peak_allocated_mb=max(prefill_peak,torch.cuda.max_memory_allocated()/1024**2),
                 peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2,output_tokens=50,cached_forwards=49)
        del cache
        if record: rows.append(row);emit('benchmark',**row)

    if cfg['mode']!='evaluation':
        for index in range(cfg['warmup']):
            emit('phase',message=f'Warm-up {index+1}/{cfg["warmup"]} (excluded from measurements)')
            iteration(0,False)
            if cancelled():return
        for index in range(cfg['repetitions']):
            emit('phase',message=f'Measuring Prefill / 50-token Decode · repetition {index+1}/{cfg["repetitions"]}')
            iteration(index+1,True)
            if cancelled():return
    evaluation_peak=None
    if rows and max(r['peak_allocated_mb'] for r in rows)>env['vram_mb']:
        raise ValueError('Measured CUDA allocation exceeds physical GPU memory. Performance receipts are retained, but this oversubscribed configuration is excluded from resident-GPU task evaluation; no accuracy result is fabricated.')
    if cfg['mode']!='benchmark':
        torch.cuda.reset_peak_memory_stats()
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        harness=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=1,max_length=2048)
        env['lm_eval']=importlib.metadata.version('lm_eval')
        for task in cfg['tasks']:
            if cancelled():return
            emit('phase',message=f'Evaluating {task} · '+('full split' if not cfg['limit'] else f'first {cfg["limit"]} examples; smoke test'))
            report=simple_evaluate(model=harness,tasks=[task],num_fewshot=0,limit=cfg['limit'] or None,
                                   log_samples=False,bootstrap_iters=0,apply_chat_template=False,
                                   random_seed=cfg['seed'],numpy_random_seed=cfg['seed'],torch_random_seed=cfg['seed'],fewshot_random_seed=cfg['seed'])
            score=task_metrics(report,task);scores.append(score);emit('task_score',**score)
        evaluation_peak=torch.cuda.max_memory_allocated()/1024**2
    medians={key:statistics.median(r[key] for r in rows) for key in ['prefill_tps','decode_tps','prefill_ms','decode_ms']} if rows else {}
    peak=max([load_peak]+[r['peak_allocated_mb'] for r in rows])
    emit('result',message='Deployment evaluation completed',model=model_id,revision=revision,
         environment=env,protocol=protocol,benchmark=medians,peak_allocated_mb=peak,
         load_peak_allocated_mb=load_peak,evaluation_peak_allocated_mb=evaluation_peak,
         task_scores=scores,attention=actual_attention,quantization=quant,adapter=cfg['adapter'],
         kernels=kernels,free_vram_before_load_mb=free_before_load,compute_dtype=str(model.dtype))
