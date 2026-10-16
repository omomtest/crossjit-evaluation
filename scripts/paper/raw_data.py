"""Verify and safely unpack the original RQ records into an ignored local cache."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def entries():
    return json.loads((ROOT / 'data/manifest.json').read_text())['archives']


def verify(entry, members=False):
    path = ROOT / 'data' / entry['path']
    if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise ValueError('archive checksum mismatch: '+entry['path'])
    sidecar = ROOT / 'data' / entry['member_manifest']
    if digest(sidecar) != entry['member_manifest_sha256']:
        raise ValueError('member manifest mismatch: '+entry['path'])
    if members:
        expected = {r['path']:r for r in json.loads(sidecar.read_text())['records']}
        with tarfile.open(path, 'r|gz') as stream:
            for member in stream:
                if member.name not in expected or not member.isfile():
                    raise ValueError('unexpected archive member: '+member.name)
                row = expected.pop(member.name)
                if member.size != row['bytes'] or hashlib.sha256(stream.extractfile(member).read()).hexdigest() != row['sha256']:
                    raise ValueError('member checksum mismatch: '+member.name)
        if expected:
            raise ValueError('missing archive records')


def unpack(entry, cache=None):
    verify(entry)
    target = Path(cache or ROOT / '.cache/raw-data') / entry['path'].removesuffix('.tar.gz')
    receipt = target / '.archive-sha256'
    if receipt.exists() and receipt.read_text().strip() == entry['sha256']:
        return target
    if target.exists():
        raise ValueError('incomplete cache; use a fresh --cache directory: '+str(target))
    target.mkdir(parents=True)
    with tarfile.open(ROOT / 'data' / entry['path'], 'r|gz') as stream:
        seen = set()
        for member in stream:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile() or member.name in seen:
                raise ValueError('unsafe archive member')
            seen.add(member.name)
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as src, destination.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    receipt.write_text(entry['sha256']+'\n')
    return target
