#!/usr/bin/env python3
"""Sequential explicit-GPU171-emotion corpus with frozen workers and per-emotion resume."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from qwen_story_pilot import ROOT, REVISIONS, git, now, save, sha, parse_gpu_ids


def corpus_config(base, original):
    if len(original['emotions']) != 171 or len(set(original['emotions'])) != 171:
        raise ValueError('Expected171 unique frozen emotion labels')
    if len(original['topics']) != 100 or len(set(original['topics'])) != 100:
        raise ValueError('Expected100 unique frozen topics')
    c=copy.deepcopy(base)
    if c.get('diversity_controls') or 'Write exactly one short English story' not in c['prompt_template']:
        raise ValueError('Expected restored single-story persistence prompt')
    if c['samples_per_combination'] != 12 or 'max_new_tokens' in c['generation']:
        raise ValueError('Expected12 independent stories/combination and no fixed token cap')
    c.update(emotions=original['emotions'],topics=original['topics'])
    return c


def emotion_config(full, index):
    c=copy.deepcopy(full);c['emotions']=[full['emotions'][index]]
    c['request_id_offset']=index*len(full['topics'])*full['samples_per_combination']
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--gpu',type=int,default=0)
    selection.add_argument('--gpus',type=str,help='Explicit GPUs on this server; balanced model parallel')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--plan',action='store_true',help='CPU-only plan; no output created or model loaded')
    args=parser.parse_args()
    if not 1<=args.batch_size<=64:parser.error('Batch size must be1–64')
    indices=list(range(171))
    gpu_ids=parse_gpu_ids(args.gpus) if args.gpus else [args.gpu]
    model_dir=args.model_dir.resolve()
    if args.plan:
        base=json.loads((ROOT/'configs/pilot.json').read_text()); original=json.loads((ROOT/'configs/emotions-topics.json').read_text())
        full=corpus_config(base,original)
        print(json.dumps(dict(emotions=len(indices),stories=1200*len(indices),global_emotion_indices=indices,length_policy="EOS_or_context_capacity",physical_gpus=gpu_ids,model_dir=str(model_dir)),indent=2));return
    out=args.output.resolve()
    if args.resume:
        meta=json.loads((out/'run.json').read_text())
        if (gpu_ids!=meta['physical_gpus'] or args.batch_size!=meta['batch_limit'] or str(model_dir)!=meta['model_dir']):
            raise ValueError('Resume GPU/batch settings differ')
        full=json.loads((out/'corpus-config.json').read_text())
        for name,digest in meta['frozen_sha256'].items():
            if sha((out/'source_snapshot'/name).read_bytes())!=digest:raise ValueError('Frozen source changed')
    else:
        base_path=ROOT/'configs/pilot.json';original_path=ROOT/'configs/emotions-topics.json'
        full=corpus_config(json.loads(base_path.read_text()),json.loads(original_path.read_text()))
        out.mkdir(parents=True,exist_ok=False)
        if shutil.disk_usage(out).free < 10*1024**3:raise RuntimeError('Less than10GiB disk headroom')
        snap=out/'source_snapshot';snap.mkdir();(out/'configs').mkdir();(out/'logs').mkdir()
        source_paths=[Path(__file__),ROOT/'scripts/qwen_story_pilot.py',ROOT/'scripts/generate_qwen38_pilot.py',base_path,original_path,ROOT/'model-assets/manifest.json']
        for path in source_paths:shutil.copyfile(path,snap/path.name)
        # The frozen worker imports the frozen module but keeps the real project root.
        bootstrap="""from pathlib import Path
import sys
snap=Path(__file__).resolve().parent
sys.path.insert(0,str(snap))
import qwen_story_pilot as runner
runner.ROOT=Path(%r)
runner.ENTRYPOINT_SOURCES=[snap/'generate_qwen38_pilot.py',Path(__file__).resolve()]
runner.main('Qwen3.8-27B')
""" % str(ROOT)
        (snap/'worker_entry.py').write_text(bootstrap)
        save(out/'corpus-config.json',full)
        for index in range(171):save(out/'configs'/f'emotion-{index:03d}.json',emotion_config(full,index))
        total=len(indices)*1200
        estimate=dict(prior_pilot_steady16_stories_per_minute=19.425527256137325,prior_pilot_max_new_tokens=512,practical_single_gpu_days_512=[8,10],note='Prior512-token pilot only; uncapped generation runtime is unmeasured; no pilot files needed')
        save(out/'estimate.json',estimate)
        meta=dict(status='created',created_at=now(),total_stories=total,total_emotions=len(indices),global_emotions=171,
                  topics_per_emotion=100,stories_per_combination=12,model='Qwen/Qwen3.8-27B',
                  revision=REVISIONS['Qwen3.8-27B'],physical_gpus=gpu_ids,batch_limit=args.batch_size,
                  model_dir=str(model_dir),length_policy="EOS_or_context_capacity",global_emotion_indices=indices,
                  dtype='bfloat16',enable_thinking=False,git_commit=git('rev-parse','HEAD'),
                  git_status=git('status','--porcelain'),python=sys.version,executable=sys.executable,
                  seed_rule='SHA256(42:global_request_id)[:8] hex; actual batch RNG seeded by first task',
                  global_task_order='emotion_id*1200 + topic_id*12 + sample_id-1',
                  frozen_sha256={p.name:sha(p.read_bytes()) for p in snap.iterdir()},
                  estimate=estimate,completed_emotions=0,completed_stories=0,emotions={},
                  resource_policy='Explicit GPUs on one server; balanced layer split when multiple;62GiB resource budget; adaptive2/4/8/16; <=5 resource retries per emotion',
                  quality_policy='All raw outputs retained. No quality retries/deletions; all text retained; no fixed output cap; model context boundary flagged.')
        save(out/'run.json',meta)
    lock=(out/'.supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=False;child=None
    def request_stop(*_):
        nonlocal stop
        stop=True
        if child is not None and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        for index in indices:
            emotion=full['emotions'][index]
            folder=out/f'emotion-{index:03d}';config_path=out/'configs'/f'emotion-{index:03d}.json'
            if json.loads(config_path.read_text())!=emotion_config(full,index):raise ValueError('Emotion config changed')
            worker_meta=folder/'run.json'
            if worker_meta.exists() and json.loads(worker_meta.read_text())['status']=='finished':
                state=json.loads(worker_meta.read_text())
            else:
                retries=0
                while True:
                    if stop:raise InterruptedError('Supervisor stopped')
                    cmd=[sys.executable,'-B','-u',str(out/'source_snapshot/worker_entry.py'),
                         '--model-dir',str(model_dir),'--config',str(config_path),'--output',str(folder),'--gpus',','.join(map(str,gpu_ids)),
                         '--batch-size',str(args.batch_size),'--auto-batch','--wait-for-gpu']
                    if folder.exists():cmd.append('--resume')
                    meta.update(status='running',current_emotion_id=index,current_emotion=emotion,updated_at=now())
                    with (out/'logs'/f'emotion-{index:03d}.log').open('ab',buffering=0) as log:
                        child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                               env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                        meta.update(child_pid=child.pid);save(out/'run.json',meta)
                        while child.poll() is None:
                            if worker_meta.exists():
                                state=json.loads(worker_meta.read_text())
                                meta.update(status='waiting_for_gpu' if state['status']=='waiting_for_gpu' else 'running',
                                            current_emotion_completed=state['completed_requests'],
                                            completed_stories=1200*len(meta['emotions'])+state['completed_requests'],updated_at=now())
                                save(out/'run.json',meta)
                            try:child.wait(timeout=10)
                            except subprocess.TimeoutExpired:pass
                    if stop:raise InterruptedError('Supervisor stopped')
                    state=json.loads(worker_meta.read_text()) if worker_meta.exists() else {}
                    if child.returncode==0 and state.get('status')=='finished':break
                    error=state.get('error','worker failed before metadata')
                    resource_error=any(s in error for s in ['OutOfMemoryError','GPU resource changed','nvidia-smi failed'])
                    if not resource_error or retries>=5:raise RuntimeError(f'emotion{index} worker exit{child.returncode}: {error}')
                    retries+=1;meta.update(status='resource_retry',resource_retry=retries,last_resource_error=error,updated_at=now());save(out/'run.json',meta)
                    print(now(),'resource retry',index,retries,error,flush=True)
                    for _ in range(6):
                        if stop:raise InterruptedError('Supervisor stopped')
                        time.sleep(10)
            if state['completed_requests']!=1200:raise RuntimeError('Unexpected per-emotion request count')
            meta['emotions'][str(index)]=dict(emotion=emotion,status='finished',stories=1200,summary=state['summary'])
            meta.update(completed_emotions=len(meta['emotions']),completed_stories=1200*len(meta['emotions']),updated_at=now(),child_pid=None)
            generation=sum(e['summary']['generation_seconds'] for e in meta['emotions'].values())
            meta['observed_generation_seconds']=generation
            meta['observed_remaining_seconds_excluding_wait']=generation/meta['completed_stories']*(meta['total_stories']-meta['completed_stories'])
            save(out/'run.json',meta)
            print(now(),f'emotion{index} {emotion} finished; {meta["completed_stories"]}/{meta["total_stories"]}',flush=True)
        meta.update(status='finished',ended_at=now(),child_pid=None);save(out/'run.json',meta)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate();child.wait()
        meta.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'error',
                    error=f'{type(error).__name__}: {error}',updated_at=now(),child_pid=None)
        save(out/'run.json',meta);raise


if __name__=='__main__':main()
