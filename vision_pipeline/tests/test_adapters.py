import numpy as np
import pytest

from vision_pipeline.adapters import (
    bigearthnet_to_imagery,
    bitemporal_array_to_imagery,
    bitemporal_optical_array_to_imagery,
    optical_sar_pair_to_imagery,
)


def test_bitemporal_array_to_imagery():
    data = np.zeros((6, 2, 2), dtype=np.float32)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = -12.0
    data[5] = -18.0

    imagery = bitemporal_array_to_imagery(data)

    assert np.array_equal(imagery["optical"]["NIR"], data[0])
    assert np.array_equal(imagery["optical"]["RED"], data[1])
    assert np.array_equal(imagery["optical"]["GREEN"], data[2])
    assert np.array_equal(imagery["optical"]["BLUE"], data[3])
    assert np.array_equal(imagery["sar"]["VV"], data[4])
    assert np.array_equal(imagery["sar"]["VV_pre"], data[4])
    assert np.array_equal(imagery["sar"]["VV_post"], data[5])


def test_bitemporal_array_to_imagery_rejects_bad_shape():
    with pytest.raises(ValueError):
        bitemporal_array_to_imagery(np.zeros((5, 2, 2), dtype=np.float32))


def test_optical_sar_pair_to_imagery():
    data = np.zeros((5, 2, 2), dtype=np.float32)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = -2.0

    imagery = optical_sar_pair_to_imagery(data)

    assert np.array_equal(imagery["optical"]["NIR"], data[0])
    assert np.array_equal(imagery["optical"]["RED"], data[1])
    assert np.array_equal(imagery["optical"]["GREEN"], data[2])
    assert np.array_equal(imagery["optical"]["BLUE"], data[3])
    assert np.array_equal(imagery["sar"]["VV"], data[4])


def test_optical_sar_pair_to_imagery_rejects_bad_shape():
    with pytest.raises(ValueError):
        optical_sar_pair_to_imagery(np.zeros((6, 2, 2), dtype=np.float32))


def test_bigearthnet_to_imagery_optical():
    data = np.zeros((12, 2, 2), dtype=np.float32)
    data[1] = 1.0  # B02 -> BLUE
    data[2] = 2.0  # B03 -> GREEN
    data[3] = 3.0  # B04 -> RED
    data[7] = 4.0  # B08 -> NIR

    imagery = bigearthnet_to_imagery(data)

    assert np.array_equal(imagery["optical"]["BLUE"], data[1])
    assert np.array_equal(imagery["optical"]["GREEN"], data[2])
    assert np.array_equal(imagery["optical"]["RED"], data[3])
    assert np.array_equal(imagery["optical"]["NIR"], data[7])
    assert "sar" not in imagery


def test_bigearthnet_to_imagery_fused():
    data = np.zeros((14, 2, 2), dtype=np.float32)
    data[1] = 1.0
    data[2] = 2.0
    data[3] = 3.0
    data[7] = 4.0
    data[12] = -10.0  # VV
    data[13] = -15.0  # VH

    imagery = bigearthnet_to_imagery(data)

    assert np.array_equal(imagery["optical"]["NIR"], data[7])
    assert np.array_equal(imagery["sar"]["VV"], data[12])
    assert np.array_equal(imagery["sar"]["VH"], data[13])


def test_bigearthnet_to_imagery_rejects_bad_shape():
    with pytest.raises(ValueError):
        bigearthnet_to_imagery(np.zeros((6, 2, 2), dtype=np.float32))


def test_bitemporal_optical_array_to_imagery():
    data = np.zeros((4, 2, 2), dtype=np.float32)
    data[0] = 1.0  # NIR_t1
    data[1] = 2.0  # RED_t1
    data[2] = 3.0  # NIR_t2
    data[3] = 4.0  # RED_t2

    imagery = bitemporal_optical_array_to_imagery(data)

    assert np.array_equal(imagery["optical"]["t1"]["NIR"], data[0])
    assert np.array_equal(imagery["optical"]["t1"]["RED"], data[1])
    assert np.array_equal(imagery["optical"]["t2"]["NIR"], data[2])
    assert np.array_equal(imagery["optical"]["t2"]["RED"], data[3])


def test_bitemporal_optical_array_to_imagery_rejects_bad_shape():
    with pytest.raises(ValueError):
        bitemporal_optical_array_to_imagery(np.zeros((6, 2, 2), dtype=np.float32))
