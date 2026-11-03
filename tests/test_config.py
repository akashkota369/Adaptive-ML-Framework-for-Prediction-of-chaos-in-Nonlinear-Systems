"""Safe configuration loading and canonical content hashing."""

import pytest

from chaosnet.configs.loader import config_hash, load_cfg


def test_config_hash_ignores_yaml_order_and_formatting(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("seed: 42\noptions: {a: 1, b: 2}\n", encoding="utf-8")
    second.write_text("# Same values\noptions:\n  b: 2\n  a: 1\nseed: 42\n", encoding="utf-8")
    one, two = load_cfg(first), load_cfg(second)
    assert one == two == {"seed": 42, "options": {"a": 1, "b": 2}}
    assert config_hash(one) == config_hash(two)
    assert config_hash(one) != config_hash({**one, "seed": 43})


@pytest.mark.parametrize("content", ["", "- a\n- b\n", "seed: .nan\n"])
def test_config_rejects_invalid_data(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_cfg(path)


def test_config_rejects_nonstring_mapping_keys(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("settings: {1: value}\n", encoding="utf-8")
    with pytest.raises(TypeError, match="keys must be strings"):
        load_cfg(path)
