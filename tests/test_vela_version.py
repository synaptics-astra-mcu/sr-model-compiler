"""Tests the vela installation"""

# import pytest
from sr_model_compiler import call_shell_cmd


def test_compiler():
    """Tests the basic compiler version"""

    success, version = call_shell_cmd("vela --version")
    print(f"Run vela --version, found {success}:{version}")

    assert success is True, "Failed to run vela command"
    assert version.rstrip() == "4.3.0", "Failed to run vela and get correct version"
