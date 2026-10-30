"""CPU preflight; optional read-only GPU query. Never load model weights."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
from model_files import verify
from qwen_story_pilot import ROOT, tasks, available_gpu, parse_gpu_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--check-gpu', type=int)
    g.add_argument('--check-gpus', type=str)
    a = p.parse_args()
    c = json.loads((ROOT/'configs/pilot.json').read_text())
    full = json.loads((ROOT/'configs/emotions-topics.json').read_text())
    assert len(full['emotions']) == len(set(full['emotions'])) == 171
    assert len(full['topics']) == len(set(full['topics'])) == 100
    assert 'max_new_tokens' not in c['generation'] and 'max_length' not in c['generation']
    assert c['samples_per_combination'] == 12 and c['enable_thinking'] is False
    count = verify(a.model_dir.resolve())
    versions = {n: md.version(n) for n in ['torch','transformers','triton','tokenizers','accelerate','safetensors','huggingface-hub','numpy']}
    if versions['torch'] != '2.7.1+cu126' or versions['transformers'] != '5.3.0':
        raise RuntimeError(f'Runtime differs from tested versions: {versions}')
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    t = tasks(c)[0]
    rendered = tok.apply_chat_template([{'role':'system','content':c['prompt_template'].format(**t)},
                                       {'role':'user','content':''}],tokenize=False,
                                      add_generation_prompt=True,enable_thinking=False)
    assert rendered.rstrip().endswith('</think>')
    model_config=json.loads((a.model_dir/'config.json').read_text())
    capacity=model_config['text_config']['max_position_embeddings']
    if a.check_gpu is not None or a.check_gpus:
        ids=parse_gpu_ids(a.check_gpus) if a.check_gpus else [a.check_gpu]
        info=available_gpu('Qwen3.8-27B',ids)
        if info is None:raise RuntimeError('Requested GPU memory/mode insufficient; no GPU was allocated')
        print(json.dumps(info,indent=2))
    print(json.dumps(dict(status='passed',files=count,total_stories=205200,context_capacity=capacity,
                          length_policy='natural EOS; context capacity guard only',dependencies=versions),indent=2))


if __name__ == '__main__':
    main()
