"""Pinned model download and integrity checks. No GPU allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def verify(directory, full_hash=False):
    entries = json.loads((ROOT / 'model-assets/manifest.json').read_text())
    for item in entries:
        path = directory / item['path']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError(f'Missing or wrong size: {path}')
        if full_hash or not path.name.endswith('.safetensors'):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                    h.update(chunk)
            if h.hexdigest() != item['sha256']:
                raise ValueError(f'SHA256 mismatch: {path}')
    return len(entries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--full-hash', action='store_true')
    a = p.parse_args()
    if a.download:
        from huggingface_hub import snapshot_download
        source = json.loads((ROOT / 'model-assets/source.json').read_text())
        entries = json.loads((ROOT / 'model-assets/manifest.json').read_text())
        a.model_dir.mkdir(parents=True, exist_ok=True)
        needed = sum(x['size'] for x in entries if not (a.model_dir/x['path']).is_file()
                     or (a.model_dir/x['path']).stat().st_size != x['size'])
        if shutil.disk_usage(a.model_dir).free < needed + 10 * 1024**3:
            raise RuntimeError('Insufficient disk: missing model bytes plus10GiB required')
        snapshot_download(repo_id=source['repo_id'], revision=source['revision'],
                          local_dir=a.model_dir, allow_patterns=[x['path'] for x in entries],
                          max_workers=4)
    count = verify(a.model_dir.resolve(), full_hash=a.full_hash)
    print(f'Verified {count} files; weight SHA256 checked: {a.full_hash}')


if __name__ == '__main__':
    main()
