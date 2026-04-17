"""Tests for CSI parser module."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.processing.csi_parser import CSIParser


class TestCSIParser:
    """Test cases for CSI data parsing."""

    def test_parser_initialization_esp32(self):
        """Test parser initializes with ESP32 format."""
        parser = CSIParser("esp32")
        assert parser.hardware == "esp32"

    def test_parser_initialization_intel5300(self):
        """Test parser initializes with Intel 5300 format."""
        parser = CSIParser("intel5300")
        assert parser.hardware == "intel5300"

    def test_parser_initialization_nexmon(self):
        """Test parser initializes with Nexmon format."""
        parser = CSIParser("nexmon")
        assert parser.hardware == "nexmon"

    def test_parser_invalid_hardware(self):
        """Test that invalid hardware format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hardware"):
            CSIParser("invalid_format")

    def test_parse_esp32_line_valid(self):
        """Test parsing a valid ESP32 CSI data line."""
        parser = CSIParser("esp32")

        # Create a minimal valid ESP32 CSI line
        # Format: CSI_DATA,<ts>,<rssi>,<rate>,<sig_mode>,<mcs>,<bw>,<smoothing>,
        #         <not_sounding>,<aggregation>,<stbc>,<fec>,<sgi>,<noise_floor>,
        #         <antenna>,<channel>,<sec_channel>,<len>,<csi_hex>,<rx_ts>
        csi_hex = "0102" * 26  # 52 subcarriers * 2 bytes (I,Q)
        line = f"CSI_DATA,1234567890,-40,1,1,7,20,0,0,0,1,0,1,-95,0,6,0,52,{csi_hex},0"

        sample = parser.parse_esp32_line(line, anchor_id="esp32_1")
        assert sample is not None
        assert sample.anchor_id == "esp32_1"
        assert sample.channel == 6
        assert sample.bandwidth == 20

    def test_parse_esp32_line_invalid(self):
        """Test that invalid lines return None."""
        parser = CSIParser("esp32")
        assert parser.parse_esp32_line("not a csi line") is None
        assert parser.parse_esp32_line("") is None

    def test_parse_nonexistent_file(self):
        """Test that parsing a nonexistent file raises FileNotFoundError."""
        parser = CSIParser("esp32")
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.csv"))