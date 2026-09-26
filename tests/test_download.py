import hashlib
import json

import pytest

from scripts import download


def test_checksum_detects_changed_content(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"original")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    download.check_file(path, digest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download.check_file(path, digest)


@pytest.mark.parametrize("name", ["../model.pt", "/model.pt", "models/../model.pt", "models\\model.pt"])
def test_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match="Invalid release path"):
        download.safe_name(name)


def test_checkpoint_uses_immutable_revision(tmp_path, monkeypatch):
    out = tmp_path / "downloads"
    model = out / "group/example/model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(json.dumps({"models": [{"id": "example", "status": "uploaded",
        "upload_path": "group/example/model.pt", "hf_revision": "immutable-commit",
        "export_sha256": hashlib.sha256(b"model").hexdigest()}]}))
    monkeypatch.setattr(download, "hf_hub_download", lambda *a, **k: str(manifest))
    calls = []
    def snapshot(*args, **kwargs):
        calls.append(kwargs)
        return str(out)
    monkeypatch.setattr(download, "snapshot_download", snapshot)
    assert download.download_checkpoint("example", out) == model
    assert calls[0]["revision"] == "immutable-commit"
    assert calls[0]["allow_patterns"] == ["group/example/*"]


def test_unpublished_checkpoint_is_not_downloaded(tmp_path, monkeypatch):
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(json.dumps({"models": [{"id": "pending", "status": "not_uploaded"}]}))
    monkeypatch.setattr(download, "hf_hub_download", lambda *a, **k: str(manifest))
    monkeypatch.setattr(download, "snapshot_download", lambda *a, **k: pytest.fail("Unexpected download"))
    with pytest.raises(ValueError, match="not yet published"):
        download.download_checkpoint("pending", tmp_path)
