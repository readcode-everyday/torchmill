"""Tests for ConfigMutator — read-modify-write for JSON and YAML configs."""

import json
import os
import tempfile

import pytest
import yaml

from torchmill.actuator.config_mutator import ConfigMutator


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def json_config(tmp_path):
    data = {"lr": 1e-3, "batch_size": 32, "optimizer": {"type": "adamw", "weight_decay": 0.01}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


@pytest.fixture
def yaml_config(tmp_path):
    data = {"lr": 1e-3, "batch_size": 32, "optimizer": {"type": "adamw", "weight_decay": 0.01}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


@pytest.fixture
def yml_config(tmp_path):
    data = {"lr": 5e-4, "epochs": 100}
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(data))
    return str(path)


# =========================================================================
# Format detection
# =========================================================================

def test_detects_json_format(json_config):
    m = ConfigMutator(json_config)
    assert m._fmt == "json"


def test_detects_yaml_format(yaml_config):
    m = ConfigMutator(yaml_config)
    assert m._fmt == "yaml"


def test_detects_yml_format(yml_config):
    m = ConfigMutator(yml_config)
    assert m._fmt == "yaml"


def test_unsupported_format_raises(tmp_path):
    bad = tmp_path / "config.toml"
    bad.write_text("lr = 1e-3\n")
    with pytest.raises(ValueError, match="Unsupported"):
        ConfigMutator(str(bad))


# =========================================================================
# Read
# =========================================================================

def test_read_json(json_config):
    m = ConfigMutator(json_config)
    data = m.read()
    assert data["lr"] == pytest.approx(1e-3)
    assert data["batch_size"] == 32
    assert data["optimizer"]["type"] == "adamw"


def test_read_yaml(yaml_config):
    m = ConfigMutator(yaml_config)
    data = m.read()
    assert data["lr"] == pytest.approx(1e-3)
    assert data["optimizer"]["weight_decay"] == pytest.approx(0.01)


# =========================================================================
# Write round-trip
# =========================================================================

def test_write_json_round_trip(json_config):
    m = ConfigMutator(json_config)
    data = m.read()
    data["lr"] = 5e-4
    m.write(data)
    data2 = m.read()
    assert data2["lr"] == pytest.approx(5e-4)


def test_write_yaml_round_trip(yaml_config):
    m = ConfigMutator(yaml_config)
    data = m.read()
    data["batch_size"] = 64
    m.write(data)
    data2 = m.read()
    assert data2["batch_size"] == 64


# =========================================================================
# apply_changes — flat keys
# =========================================================================

def test_apply_flat_key_json(json_config):
    m = ConfigMutator(json_config)
    updated = m.apply_changes({"lr": 3e-4, "batch_size": 64})
    assert updated["lr"] == pytest.approx(3e-4)
    assert updated["batch_size"] == 64
    # Verify persisted to disk
    reloaded = m.read()
    assert reloaded["lr"] == pytest.approx(3e-4)


def test_apply_flat_key_yaml(yaml_config):
    m = ConfigMutator(yaml_config)
    updated = m.apply_changes({"lr": 1e-5})
    assert updated["lr"] == pytest.approx(1e-5)
    reloaded = m.read()
    assert reloaded["lr"] == pytest.approx(1e-5)


# =========================================================================
# apply_changes — dotted (nested) keys
# =========================================================================

def test_apply_dotted_key_json(json_config):
    m = ConfigMutator(json_config)
    updated = m.apply_changes({"optimizer.weight_decay": 0.05})
    assert updated["optimizer"]["weight_decay"] == pytest.approx(0.05)
    reloaded = m.read()
    assert reloaded["optimizer"]["weight_decay"] == pytest.approx(0.05)


def test_apply_dotted_key_yaml(yaml_config):
    m = ConfigMutator(yaml_config)
    updated = m.apply_changes({"optimizer.type": "sgd", "optimizer.weight_decay": 0.001})
    assert updated["optimizer"]["type"] == "sgd"
    assert updated["optimizer"]["weight_decay"] == pytest.approx(0.001)


def test_apply_creates_nested_key(json_config):
    """Dotted key should create nested dict if it doesn't exist."""
    m = ConfigMutator(json_config)
    updated = m.apply_changes({"scheduler.type": "cosine"})
    assert updated["scheduler"]["type"] == "cosine"


# =========================================================================
# apply_changes — empty changes
# =========================================================================

def test_apply_empty_changes(json_config):
    m = ConfigMutator(json_config)
    original = m.read()
    updated = m.apply_changes({})
    assert updated == original


# =========================================================================
# Original fields preserved
# =========================================================================

def test_apply_preserves_other_keys(json_config):
    m = ConfigMutator(json_config)
    m.apply_changes({"lr": 9e-5})
    data = m.read()
    assert data["batch_size"] == 32
    assert data["optimizer"]["type"] == "adamw"
