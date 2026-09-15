#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features


def test_hw_to_dataset_features_routes_env_features_to_environment_state():
    hw = {
        "joint.pos": float,
        "tactile0.x": PolicyFeature(type=FeatureType.ENV, shape=(1,)),
        "tactile0.y": PolicyFeature(type=FeatureType.ENV, shape=(1,)),
        "cam": (480, 640, 3),
    }

    features = hw_to_dataset_features(hw, OBS_STR)

    assert features[OBS_STATE]["names"] == ["joint.pos"]
    assert features[OBS_STATE]["shape"] == (1,)
    assert features[OBS_ENV_STATE] == {
        "dtype": "float32",
        "shape": (2,),
        "names": ["tactile0.x", "tactile0.y"],
    }


def test_hw_to_dataset_features_without_env_features_adds_no_environment_state():
    features = hw_to_dataset_features({"joint.pos": float, "cam": (480, 640, 3)}, OBS_STR)

    assert features[OBS_STATE]["names"] == ["joint.pos"]
    assert OBS_ENV_STATE not in features


def test_hw_to_dataset_features_keeps_non_env_policy_features_in_state():
    hw = {
        "joint.pos": float,
        "extra": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
    }

    features = hw_to_dataset_features(hw, OBS_STR)

    assert features[OBS_STATE]["names"] == ["joint.pos", "extra"]
    assert OBS_ENV_STATE not in features


def test_hw_to_dataset_features_ignores_env_features_on_the_action_side():
    hw = {
        "joint.pos": float,
        "tactile0.x": PolicyFeature(type=FeatureType.ENV, shape=(1,)),
    }

    features = hw_to_dataset_features(hw, ACTION)

    assert features[ACTION]["names"] == ["joint.pos"]
    assert OBS_ENV_STATE not in features
