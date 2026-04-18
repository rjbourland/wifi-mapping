"""Tests for HardwareManager permission checks and scan lifecycle."""

import sys
import pytest

from gui.utils.hardware import HardwareManager


class TestCheckScanPermissions:
    def test_returns_tuple(self):
        result = HardwareManager.check_scan_permissions()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_windows_or_macos_always_permitted(self):
        """On non-Linux platforms, scanning should always be permitted."""
        if sys.platform not in ("win32", "darwin"):
            pytest.skip("Only runs on Windows/macOS")
        has_perm, msg = HardwareManager.check_scan_permissions()
        assert has_perm is True
        assert isinstance(msg, str)

    def test_linux_permission_message_includes_fix(self):
        """On Linux without permissions, message should mention setcap."""
        if sys.platform == "win32" or sys.platform == "darwin":
            pytest.skip("Only relevant on Linux")
        import os
        if os.geteuid() == 0:
            pytest.skip("Running as root — always has permissions")
        # Check if CAP_NET_RAW is set
        has_perm, msg = HardwareManager.check_scan_permissions()
        if not has_perm:
            assert "setcap" in msg


class TestHardwareManagerLifecycle:
    def test_initial_state(self):
        hw = HardwareManager()
        assert hw.rssi_scanner is None
        assert hw.is_active is False

    def test_stop_resets_state(self):
        hw = HardwareManager()
        hw._active = True
        hw.stop()
        assert hw.is_active is False
        assert hw.rssi_scanner is None

    def test_scan_rssi_returns_empty_when_not_started(self):
        hw = HardwareManager()
        result = hw.scan_rssi()
        assert result == []

    def test_scan_and_process_returns_empty_when_not_started(self):
        hw = HardwareManager()
        result = hw.scan_and_process()
        assert result == []