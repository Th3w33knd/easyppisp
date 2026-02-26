"""
Tests for validation utilities and custom exceptions.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import pytest
import torch

from easyppisp.validation import (
    PPISPShapeError,
    PPISPValueError,
    PPISPDeviceError,
    PPISPPhysicsWarning,
    check_image_shape,
    check_same_device,
    check_linear_radiance,
    check_exposure_range,
)


class TestCheckImageShape:
    def test_valid_hwc(self):
        check_image_shape(torch.ones(8, 8, 3))   # should not raise

    def test_valid_bhwc(self):
        check_image_shape(torch.ones(2, 8, 8, 3))  # batch — should not raise

    def test_1d_raises(self):
        with pytest.raises(PPISPShapeError, match="3D or 4D"):
            check_image_shape(torch.ones(10))

    def test_wrong_channel_count_raises(self):
        with pytest.raises(PPISPShapeError, match="3 channels"):
            check_image_shape(torch.ones(8, 8, 4))   # RGBA

    def test_custom_name_in_error(self):
        with pytest.raises(PPISPShapeError, match="my_tensor"):
            check_image_shape(torch.ones(10), name="my_tensor")


class TestCheckSameDevice:
    def test_same_device_ok(self):
        a = torch.zeros(3)
        b = torch.zeros(3)
        check_same_device(a, b)   # should not raise

    def test_single_tensor_ok(self):
        check_same_device(torch.zeros(3))


class TestCheckLinearRadiance:
    def test_uint8_range_warns(self):
        """Values > 10 look like uint8 and should trigger a warning."""
        img = torch.ones(4, 4, 3) * 128.0   # [0, 255] scale
        with pytest.warns(PPISPPhysicsWarning):
            check_linear_radiance(img)

    def test_uint8_range_enforced_raises(self):
        img = torch.ones(4, 4, 3) * 200.0
        with pytest.raises(PPISPValueError):
            check_linear_radiance(img, enforce=True)

    def test_linear_radiance_no_warning(self):
        """A [0, 1] float tensor should not trigger any warnings."""
        import warnings
        img = torch.rand(4, 4, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PPISPPhysicsWarning)
            check_linear_radiance(img)   # should not raise


class TestCheckExposureRange:
    def test_normal_range_no_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", PPISPPhysicsWarning)
            check_exposure_range(2.0)   # fine

    def test_extreme_value_warns(self):
        with pytest.warns(PPISPPhysicsWarning, match="unusually large"):
            check_exposure_range(15.0)


class TestExceptionHierarchy:
    def test_shape_error_is_value_error(self):
        assert issubclass(PPISPShapeError, ValueError)

    def test_value_error_is_value_error(self):
        assert issubclass(PPISPValueError, ValueError)

    def test_device_error_is_runtime_error(self):
        assert issubclass(PPISPDeviceError, RuntimeError)

    def test_physics_warning_is_user_warning(self):
        assert issubclass(PPISPPhysicsWarning, UserWarning)
