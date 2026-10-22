"""Measured single-request CUDA benchmark and optional lm-eval task evaluation."""
import hashlib
import importlib.metadata
import importlib.util
import math
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
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from nano_posttrain.utils.seed import set_seed

    if not torch.cuda.is_available():
        raise ValueError('Deployment measurements require CUDA; CPU results are not substituted')
    requirements=[]
    if cfg['mode'] != 'benchmark': requirements.append('lm_eval')
    if cfg['attention']=='flash_attention_2': requirements.append('flash_attn')
    if cfg['quantization'].startswith('gptq'): requirements.append('gptqmodel')
    if cfg['quantization']=='awq': requirements.append('awq')
    missing=[name for name in requirements if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError('Missing deployment dependencies: '+', '.join(missing)+'. Use a compatible isolated environment; no fallback was applied.')
    set_seed(cfg['seed'])
    model_id=cfg['model']+SUFFIXES[cfg['quantization']]
    emit('phase',message=f'Resolving {model_id}; {cfg["attention"]}, {cfg["dtype"]}')
    path=snapshot_download(model_id,revision=adapter.get('revision') if adapter else None,
                           local_files_only=not cfg['allow_download'],
                           allow_patterns=['*.json','*.txt','*.safetensors','*.jinja'])
    if cancelled(): return
    revision=path.replace('\\','/').split('/')[-1]
    torch.cuda.reset_peak_memory_stats()
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
    env=dict(gpu=torch.cuda.get_device_name(0),vram_mb=round(torch.cuda.get_device_properties(0).total_memory/1024**2),
             torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
             python=platform.python_version(),system=platform.system(),
             gpu_capability=list(torch.cuda.get_device_capability(0)))
    load_peak=torch.cuda.max_memory_allocated()/1024**2
    protocol=dict(batch_size=1,input_tokens=cfg['input_tokens'],decode_tokens=50,warmup=cfg['warmup'],
                  repetitions=cfg['repetitions'],seed=cfg['seed'],num_fewshot=0,apply_chat_template=False,
                  limit=cfg['limit'],evaluation_max_length=2048,
                  decode_definition='50 greedy output tokens: first from prefill logits, followed by 49 KV-cached forward passes; EOS ignored',
                  memory_definition='CUDA peak allocated/reserved includes resident weights; phases reset separately; no CPU offload')
    base_ids=tokenizer('Describe a stainless steel travel bottle priced at $24.99. Use only supplied product facts. ',add_special_tokens=False)['input_ids']
    ids=(base_ids*(cfg['input_tokens']//len(base_ids)+1))[:cfg['input_tokens']]
    protocol['prompt_sha256']=hashlib.sha256(bytes(str(ids),'utf-8')).hexdigest()
    emit('deployment_info',model=model_id,revision=revision,environment=env,protocol=protocol,
         attention=actual_attention,quantization=quant,adapter=cfg['adapter'])
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
         task_scores=scores,attention=actual_attention,quantization=quant,adapter=cfg['adapter'])
