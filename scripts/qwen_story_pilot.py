"""Explicit-GPU BF16 pilot generation, preserving raw outputs and reproducibility metadata."""
import argparse
import csv
import datetime
import fcntl
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_SOURCES = [ROOT / 'scripts/generate_qwen38_pilot.py']
REVISIONS = {
    'Qwen3.8-27B': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
}
MIN_FREE_MIB = {'Qwen3.8-27B': 62 * 1024}


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def diversity_plan(config, emotion, topic_id):
    controls = config.get('diversity_controls')
    if controls is None:
        return None
    perspectives = controls['perspective']
    intensities = controls['intensity']
    channels = controls['expression_channel']
    if (len(perspectives), len(intensities), len(channels), config['samples_per_combination']) != (2, 3, 4, 12):
        raise ValueError('Diversity pilot requires 2 perspectives, 3 intensities, 4 channels, and 12 samples')
    plan = [dict(perspective=perspectives[(level + channel) % 2], intensity=intensity,
                 expression_channel=expression)
            for level, intensity in enumerate(intensities)
            for channel, expression in enumerate(channels)]
    plan_seed = int(sha(f"{config['base_seed']}:{emotion}:{topic_id}:diversity".encode())[:8], 16)
    random.Random(plan_seed).shuffle(plan)
    return plan


def tasks(config):
    rows = []
    for emotion in config['emotions']:
        for topic_id, topic in enumerate(config['topics']):
            plan = diversity_plan(config, emotion, topic_id)
            for sample in range(1, config['samples_per_combination'] + 1):
                index = len(rows) + config.get('request_id_offset', 0)
                rows.append(dict(request_id=index, emotion=emotion, topic_id=topic_id,
                                 topic=topic, sample_id=sample,
                                 seed=int(sha(f"{config['base_seed']}:{index}".encode())[:8], 16)))
                if plan is not None:
                    rows[-1].update(plan[sample - 1])
    return rows


def parse_gpu_ids(value):
    ids = [int(x.strip()) for x in value.split(',')]
    if not ids or len(set(ids)) != len(ids) or any(x < 0 for x in ids):
        raise ValueError('GPU IDs must be distinct nonnegative integers')
    return ids


def available_gpu(model, requested):
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,utilization.gpu,compute_mode',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError('nvidia-smi failed: ' + p.stderr.strip())
    rows = []
    for row in csv.reader(io.StringIO(p.stdout), skipinitialspace=True):
        index, uuid, name, free, utilization, mode = [x.strip() for x in row]
        rows.append(dict(index=int(index), uuid=uuid, name=name, free_mib=int(free),
                         utilization_percent=int(utilization), mode=mode))
    if requested is None:
        candidates=[r for r in rows if r['mode']=='Default' and r['free_mib']>=MIN_FREE_MIB[model]]
        selected=[max(candidates,key=lambda r:r['free_mib'])] if candidates else []
    else:
        ids=requested if isinstance(requested,list) else [requested]
        table={r['index']:r for r in rows}
        if any(i not in table for i in ids):raise ValueError('Requested GPU does not exist')
        selected=[table[i] for i in ids]
    if not selected or any(r['mode']!='Default' for r in selected):return None
    if len(selected)==1:
        if selected[0]['free_mib']<MIN_FREE_MIB[model]:return None
    elif (any(r['free_mib']<6144 for r in selected)
          or sum(r['free_mib']-2048 for r in selected)<MIN_FREE_MIB[model]):return None
    return dict(index=[r['index'] for r in selected],uuid=','.join(r['uuid'] for r in selected),
                devices=selected,free_mib=sum(r['free_mib'] for r in selected),mode='Default')


def parse_body(text, tokenizer, eos):
    stripped = text.strip()
    marker_ok = stripped.startswith('<NEW STORY>') and stripped.count('<NEW STORY>') == 1
    body = stripped.removeprefix('<NEW STORY>').strip()
    length = len(tokenizer.encode(body, add_special_tokens=False))
    return body, length, marker_ok


def trim_at_eos(ids, eos_ids):
    """Keep the first EOS, discard batch padding after it."""
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[:index + 1], True
    return ids, False


class BatchSizer:
    def __init__(self, maximum, adaptive):
        self.maximum = maximum
        self.adaptive = adaptive
        self.current = min(2, maximum) if adaptive else maximum
        self.growth_enabled = adaptive

    def succeeded(self):
        if self.growth_enabled:
            self.current = min(self.maximum, self.current * 2)

    def failed(self, attempted):
        if attempted <= 1:
            return False
        self.current = max(1, attempted // 2)
        self.growth_enabled = False
        return True


def finalize(out, meta, records, all_tasks, model_name):
    summary = dict(stories=len(records), length_in_range=sum(r['body_token_range_ok'] for r in records),
                   marker_ok=sum(r['marker_ok'] for r in records),
                   truncated=sum(r['truncated'] for r in records),
                   target_word_warnings=sum(r['target_word_warning'] for r in records),
                   generation_seconds=sum(r['generation_seconds'] for r in records),
                   generated_tokens=sum(r['generated_tokens'] for r in records),
                   quality_scope='mechanical checks only; synonyms and semantic fidelity need review')
    save(out / 'summary.json', summary)
    sections = [f'# {model_name} — {len(all_tasks)} stories\n']
    for row in records:
        sections.append(f'## {row["emotion"]} / Topic {row["topic_id"] + 1} / Sample {row["sample_id"]}\n\n'
                        f'{row["topic"]}\n\n{row["body"]}\n\n'
                        f'Body tokens: {row["body_tokens"]}; seed: {row["seed"]}\n')
    (out / 'stories.md').write_text('\n'.join(sections))
    meta.update(status='finished', completed_requests=len(records), ended_at=now(), summary=summary,
                peak_gpu_allocated_bytes=max((r.get('batch_peak_gpu_allocated_bytes', 0) for r in records), default=0))
    save(out / 'run.json', meta)


def main(model_name):
    parser = argparse.ArgumentParser(description=__doc__)
    default_name = 'qwen38-pilot'
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / default_name)
    parser.add_argument('--model-dir', type=Path, required=True)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--gpu', type=int, help='One physical GPU index')
    selection.add_argument('--gpus', type=str, help='Explicit physical GPU IDs, e.g.1,2; split one model, not replicas')
    parser.add_argument('--wait-for-gpu', action='store_true', help='Wait, checking once per minute')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pilot.json')
    parser.add_argument('--batch-size', type=int, default=1, help='Maximum simultaneous stories')
    parser.add_argument('--auto-batch', action='store_true', help='Grow from batch2; shrink on OOM')
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 64:
        parser.error('--batch-size must be between 1 and 64')
    gpu_ids=parse_gpu_ids(args.gpus) if args.gpus else ([args.gpu] if args.gpu is not None else None)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    min_body_tokens, max_body_tokens = config['body_token_range']
    if not (isinstance(min_body_tokens, int) and isinstance(max_body_tokens, int)
            and 0 < min_body_tokens <= max_body_tokens):
        raise ValueError('Invalid body_token_range')
    all_tasks = tasks(config)
    model_path = args.model_dir.resolve()
    signature_sources = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha(p.read_bytes())
                         for p in [Path(__file__), *ENTRYPOINT_SOURCES, config_path, ROOT / 'model-assets/manifest.json']}
    batching = dict(maximum=args.batch_size, adaptive=args.auto_batch)
    signature = sha(json.dumps(dict(config=config, batching=batching, gpu_ids=gpu_ids, model=model_name, model_path=str(model_path), revision=REVISIONS[model_name],
                                    sources=signature_sources), sort_keys=True).encode())
    out = args.output.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Use a new output directory or --resume: {out}')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    versions = {n: importlib.metadata.version(n) for n in [
        'torch', 'transformers', 'tokenizers', 'accelerate', 'safetensors', 'huggingface-hub']}
    if args.resume:
        meta = json.loads((out / 'run.json').read_text())
        if meta['signature'] != signature or meta['dependencies'] != versions:
            raise ValueError('Resume configuration/source/dependency mismatch')
    else:
        meta = dict(status='created', created_at=now(), signature=signature,
                    config=config, model='Qwen/' + model_name, revision=REVISIONS[model_name],
                    model_path=str(model_path), dependencies=versions,
                    python=sys.version, executable=sys.executable,
                    git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                    total_requests=len(all_tasks), completed_requests=0,
                    dtype='bfloat16', batch_size=args.batch_size, batching=batching,
                    seed_rule='task seed=SHA256(base_seed:request_id)[:8] hex; actual batch_seed=first task seed',
                    random_sampling_note='Batch size/composition affects sampling; task seeds are identifiers, not independent per-row RNG streams.',
                    enable_thinking=False, extraction_and_injection='not applicable',
                    length_policy='EOS or actual model context capacity; no fixed output cap',
                    token_counting='own tokenizer; strip leading story marker; no special tokens',
                    quality_policy='retain all raw outputs; flag issues; no quality retries',
                    source_sha256=signature_sources, required_free_mib=MIN_FREE_MIB[model_name])
        meta['git_dirty'] = bool(meta['git_status'])
        save(out / 'run.json', meta)
        snapshot = out / 'source_snapshot'
        snapshot.mkdir()
        for name in signature_sources:
            shutil.copyfile(ROOT / name, snapshot / Path(name).name)
        save(out / 'tasks.json', all_tasks)
    records = []
    data_file = out / 'stories.jsonl'
    if data_file.exists():
        # Recover only an unterminated final write; validate complete committed rows.
        raw = data_file.read_bytes()
        if raw and not raw.endswith(b'\n'):
            retained = raw[:raw.rfind(b'\n') + 1]
            with data_file.open('r+b') as stream:
                stream.truncate(len(retained))
            raw = retained
        records = [json.loads(line) for line in raw.splitlines()]
        for i, row in enumerate(records):
            if i >= len(all_tasks) or any(row.get(k) != v for k, v in all_tasks[i].items()):
                raise ValueError('Resume record/task mismatch')
    try:
        if len(records) == len(all_tasks):
            finalize(out, meta, records, all_tasks, model_name)
            return
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB disk headroom')
        verification_path = ROOT / 'model-assets/manifest.json'
        verified = json.loads(verification_path.read_text())
        for entry in verified:
            path = model_path / entry['path']
            if not path.is_file() or path.stat().st_size != entry['size']:
                raise RuntimeError('Model file missing or size changed: ' + str(path))
        meta['download_verification_sha256'] = sha(verification_path.read_bytes())
        while True:
            gpu = available_gpu(model_name, gpu_ids)
            if gpu:
                break
            meta.update(status='waiting_for_gpu', updated_at=now(), completed_requests=len(records))
            save(out / 'run.json', meta)
            if not args.wait_for_gpu:
                raise RuntimeError('Selected GPUs have insufficient free memory; use --wait-for-gpu')
            print(f'{now()} waiting for >= {MIN_FREE_MIB[model_name]} MiB resource budget on selected GPUs', flush=True)
            time.sleep(60)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
        # Temp/cache locations are controlled by the caller; never use a source-server path.
        import torch
        import numpy as np
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        torch.set_num_threads(4)
        if torch.cuda.device_count() != len(gpu['devices']):
            raise RuntimeError('GPU resource changed before model loading')
        free_bytes=[torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        required=MIN_FREE_MIB[model_name]*1024**2
        if (len(free_bytes)==1 and free_bytes[0]<required) or (len(free_bytes)>1 and (min(free_bytes)<6*1024**3 or sum(x-2*1024**3 for x in free_bytes)<required)):
            raise RuntimeError('GPU resource changed before model loading')
        max_memory={i:max(0,n-2*1024**3) for i,n in enumerate(free_bytes)}
        max_memory['cpu']=0
        meta['parallelism']='single_gpu' if len(free_bytes)==1 else 'balanced_layer_model_parallel'
        meta['gpu_max_memory_bytes']=max_memory
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.padding_side = 'left'
        meta['padding_side'] = 'left'
        template = tokenizer.chat_template
        if not template:
            raise ValueError('Missing tokenizer chat template')
        (out / 'chat_template.jinja').write_text(template)
        meta.update(status='loading_model', gpu=gpu, updated_at=now(), chat_template_sha256=sha(template.encode()))
        save(out / 'run.json', meta)
        print(f'{now()} loading {model_name} on physical GPU {gpu["index"]}', flush=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map={'': 'cuda:0'} if len(free_bytes)==1 else 'balanced',
            max_memory=max_memory, local_files_only=True, attn_implementation='sdpa')
        model.eval()
        context_capacity = int(model.config.text_config.max_position_embeddings)
        meta['context_capacity'] = context_capacity
        meta['actual_model_dtype'] = str(next(model.parameters()).dtype)
        device_map = getattr(model, 'hf_device_map', {'': next(model.parameters()).device})
        meta['actual_device_map'] = {k: str(v) for k, v in device_map.items()}
        if any(p.device.type!='cuda' or p.device.index>=len(free_bytes) for p in model.parameters()):
            raise RuntimeError('Expected all weights on explicitly selected GPUs; CPU/disk offload forbidden')
        input_device=model.get_input_embeddings().weight.device
        def synchronize():
            for i in range(len(free_bytes)):torch.cuda.synchronize(i)
        meta['generation_base_config'] = model.generation_config.to_dict()
        meta['attention_implementation'] = 'sdpa'
        meta['gpu_name'] = torch.cuda.get_device_name(0)
        meta.update(status='running', started_at=now())
        save(out / 'run.json', meta)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        sizer = BatchSizer(args.batch_size, args.auto_batch)
        if args.resume and 'batch_sizer' in meta:
            sizer.current = meta['batch_sizer']['current']
            sizer.growth_enabled = meta['batch_sizer']['growth_enabled']
        batch_index = max((r.get('batch_index', -1) for r in records), default=-1) + 1
        meta.setdefault('batch_history', [])
        meta.setdefault('oom_events', [])
        with data_file.open('a') as log:
            while len(records) < len(all_tasks):
                batch = all_tasks[len(records):len(records) + sizer.current]
                prompts, batch_messages = [], []
                for task in batch:
                    prompt = config['prompt_template'].format(**task)
                    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': ''}]
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                             add_generation_prompt=True, enable_thinking=False)
                    if not rendered.rstrip().endswith('</think>'):
                        raise ValueError('Non-thinking template suffix not present')
                    prompts.append(rendered)
                    batch_messages.append(messages)
                batch_seed = batch[0]['seed']
                random.seed(batch_seed)
                np.random.seed(batch_seed)
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)
                synchronize()
                for i in range(len(free_bytes)):torch.cuda.reset_peak_memory_stats(i)
                started = time.perf_counter()
                inputs, result = None, None
                try:
                    inputs = tokenizer(prompts, return_tensors='pt', add_special_tokens=False,
                                       padding=True).to(input_device)
                    with torch.inference_mode():
                        capacity = context_capacity - inputs.input_ids.shape[1]
                        if capacity <= 0:
                            raise ValueError('Prompt exceeds model context capacity')
                        params = dict(config['generation'])
                        if 'max_new_tokens' in params or 'max_length' in params:
                            raise ValueError('Fixed token caps are disabled in this package')
                        # Transformers needs a finite loop guard: use the model context boundary,
                        # not its default max_length=20 or a512/2048 story cutoff.
                        result = model.generate(**inputs, **params, max_new_tokens=capacity)
                    synchronize()
                except torch.cuda.OutOfMemoryError:
                    inputs, result = None, None
                    gc.collect()
                    for i in range(len(free_bytes)):
                        with torch.cuda.device(i):torch.cuda.empty_cache()
                    meta['oom_events'].append(dict(at=now(), first_request_id=batch[0]['request_id'],
                                                   batch_size=len(batch)))
                    if not sizer.failed(len(batch)):
                        save(out / 'run.json', meta)
                        raise
                    meta.update(active_batch_size=sizer.current, updated_at=now())
                    meta['batch_sizer'] = dict(current=sizer.current, growth_enabled=sizer.growth_enabled)
                    save(out / 'run.json', meta)
                    print(f'{now()} OOM batch{len(batch)}; retry uncommitted requests at batch{sizer.current}', flush=True)
                    continue
                elapsed = time.perf_counter() - started
                peak = sum(torch.cuda.max_memory_allocated(i) for i in range(len(free_bytes)))
                output_ids = result[:, inputs.input_ids.shape[1]:].tolist()
                input_lengths = inputs.attention_mask.sum(dim=1).tolist()
                padded_input_length = inputs.input_ids.shape[1]
                del inputs, result
                batch_records = []
                for position, (task, ids) in enumerate(zip(batch, output_ids)):
                    ids, ended = trim_at_eos(ids, eos_ids)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    body, length, marker_ok = parse_body(text, tokenizer, eos_ids)
                    row = dict(**task, text=text, body=body, body_tokens=length,
                               body_token_range_ok=min_body_tokens <= length <= max_body_tokens, marker_ok=marker_ok,
                               ended_with_eos=ended, truncated=not ended,
                               termination_reason='eos' if ended else 'model_context_capacity',
                               target_word_warning=bool(re.search(r'\b' + re.escape(task['emotion']) + r'\b', body, re.I)),
                               generated_tokens=len(ids), generation_seconds=elapsed / len(batch),
                               batch_generation_seconds=elapsed, batch_size=len(batch), batch_seed=batch_seed,
                               batch_position=position, batch_index=batch_index, batch_peak_gpu_allocated_bytes=peak,
                               input_tokens=int(input_lengths[position]), padded_input_tokens=padded_input_length,
                               messages=batch_messages[position], rendered_prompt_sha256=sha(prompts[position].encode()),
                               completed_at=now())
                    batch_records.append(row)
                for row in batch_records:
                    log.write(json.dumps(row, ensure_ascii=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
                records.extend(batch_records)
                meta['batch_history'].append(dict(batch_index=batch_index, first_request_id=batch[0]['request_id'],
                                                 batch_size=len(batch), batch_seed=batch_seed, seconds=elapsed,
                                                 peak_gpu_allocated_bytes=peak, stories_per_minute=len(batch)*60/elapsed))
                sizer.succeeded()
                meta['batch_sizer'] = dict(current=sizer.current, growth_enabled=sizer.growth_enabled)
                meta.update(completed_requests=len(records), updated_at=now(), active_batch_size=len(batch))
                save(out / 'run.json', meta)
                print(f'{now()} {len(records)}/{len(all_tasks)} batch={len(batch)} '
                      f'seconds={elapsed:.2f} stories/min={len(batch)*60/elapsed:.2f} peak_GiB={peak/1024**3:.2f}', flush=True)
                batch_index += 1
        finalize(out, meta, records, all_tasks, model_name)
    except BaseException as error:
        meta.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                    error=f'{type(error).__name__}: {error}', updated_at=now(), completed_requests=len(records))
        save(out / 'run.json', meta)
        raise
