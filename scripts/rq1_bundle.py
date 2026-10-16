"""Portable RQ1 evidence bundles with per-file hashes and restricted extraction."""
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from common import digest, read_json, save


def bundle(run, output, machine, run_id):
    run, output = Path(run), Path(output)
    files = {}
    for name in ("experiment", "metadata", "evaluation"):
        directory = run / name
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_symlink():
                    raise ValueError("bundle cannot contain symlinks: " + str(path))
                if path.is_file():
                    files[path.relative_to(run).as_posix()] = path
    manifest = dict(schema_version=1, machine=machine, run_id=run_id,
                    files={name: dict(sha256=digest(path), size=path.stat().st_size)
                           for name, path in files.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        index = temporary / "bundle.json"
        save(index, manifest)
        archive = temporary / "bundle.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(index, arcname="bundle.json", recursive=False)
            for name, path in files.items():
                tar.add(path, arcname=name, recursive=False)
        archive.replace(output)
    output.with_name(output.name + ".sha256").write_text(digest(output) + "  " + output.name + "\n")
    return manifest


def unpack(archive, output):
    """Verify before publishing; never overwrite another run or restore links."""
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError("collection destination already exists: " + str(output))
    checksum = archive.with_name(archive.name + ".sha256")
    if not checksum.is_file() or checksum.read_text().split()[0] != digest(archive):
        raise ValueError("archive SHA-256 missing or mismatched: " + str(archive))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = set()
            for member in members:
                path = PurePosixPath(member.name)
                if (not member.isfile() or path.is_absolute() or ".." in path.parts
                        or "\\" in member.name or path.as_posix() != member.name
                        or member.name in names):
                    raise ValueError("unsafe or duplicate archive member: " + member.name)
                names.add(member.name)
            if "bundle.json" not in names:
                raise ValueError("bundle index missing")
            index_member = tar.getmember("bundle.json")
            if index_member.size > 16 * 1024 * 1024:
                raise ValueError("bundle index too large")
            index = json.load(tar.extractfile(index_member))
            if index.get("schema_version") != 1 or set(index["files"]) != names - {"bundle.json"}:
                raise ValueError("bundle file inventory mismatch")
            for member in members:
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                if member.name != "bundle.json":
                    row = index["files"][member.name]
                    if target.stat().st_size != row["size"] or digest(target) != row["sha256"]:
                        raise ValueError("bundle file hash mismatch: " + member.name)
        stage.rename(output)
    return read_json(output / "bundle.json")
