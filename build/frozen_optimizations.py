"""Install the complete measured v32 configuration without changing its evidence."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "config/optimizations/v32"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bundle(bundle=BUNDLE):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "bundle.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("preset") != "v32":
        raise ValueError("unsupported frozen optimization bundle")
    for relative, record in manifest["files"].items():
        path = bundle / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(bundle):
            raise ValueError("unsafe frozen optimization path: " + relative)
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError("frozen optimization evidence drift: " + relative)
    for name, field in (("profile.json", "source_evidence"),
                        ("production_selection.json", "measurement_inputs")):
        for evidence in json.loads((bundle / name).read_text())[field]:
            relative = manifest["evidence_paths"][evidence["path"]]
            if manifest["files"][relative]["sha256"] != evidence["sha256"]:
                raise ValueError("frozen optimization evidence binding mismatch")
    native = json.loads((bundle / "native_policy.json").read_text())
    for row in native["records"]:
        evidence = row["evidence"]
        relative = manifest["evidence_paths"][evidence["path"]]
        if manifest["files"][relative]["sha256"] != evidence["sha256"]:
            raise ValueError("native evidence binding mismatch")
    if sum(r["decision"] == "ACCEPT" for r in native["records"]) != manifest["native_accepted_plan_count"]:
        raise ValueError("native selection count drift")
    return manifest


def install_frozen(source, bundle=BUNDLE):
    """Relocate evidence paths, preserving every measured byte and decision."""
    bundle = Path(bundle).resolve()
    manifest = validate_bundle(bundle)
    destination = Path(source).resolve() / "evaluation-build-inputs/v32"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in manifest["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / relative, target)
    native = json.loads((bundle / "native_policy.json").read_text())
    for row in native["records"]:
        evidence = row["evidence"]
        evidence["path"] = str(destination / manifest["evidence_paths"][evidence["path"]])
    # The original native policy is retained unchanged in the committed bundle.
    # Only file locations change here; measurement hashes and decisions do not.
    (destination / "native_policy.json").write_text(json.dumps(native, indent=2, sort_keys=True) + "\n")
    return destination


def meson_flags(directory):
    return ["-Dcrossjit-affine-profile=" + str(directory / "profile.json"),
            "-Dcrossjit-production-selection=" + str(directory / "production_selection.json"),
            "-Dcrossjit-execution-policy=" + str(directory / "execution_decision_db.json"),
            "-Dcrossjit-native-policy=" + str(directory / "native_policy.json")]


def identity(source, bundle=BUNDLE):
    manifest = validate_bundle(bundle)
    installed = Path(source) / "evaluation-build-inputs/v32"
    return dict(preset="v32", bundle_sha256=digest(Path(bundle) / "bundle.json"),
                measurement_origin=manifest["measurement_origin"],
                native_accepted_plan_count=manifest["native_accepted_plan_count"],
                expected_S_stress1_activity=manifest["expected_S_stress1_activity"],
                original_hashes={name: record["sha256"] for name, record in manifest["files"].items()},
                installed_policy_hashes={name: digest(installed / name) for name in (
                    "profile.json", "production_selection.json", "execution_decision_db.json", "native_policy.json")})
