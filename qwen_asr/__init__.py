# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
"""
qwen_asr: Qwen3-ASR package.
"""
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

from .inference.qwen3_asr import Qwen3ASRModel
from .inference.qwen3_forced_aligner import Qwen3ForcedAligner

from .inference.utils import parse_asr_output

__all__ = ["__version__"]

def register_qwen3_asr():
    """注册 qwen3_asr 架构到 transformers 5.x"""
    try:
        from .core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration
        )
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModelForSpeechSeq2Seq.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    except Exception as e:
        print(f"注册 qwen3_asr 失败: {e}")

register_qwen3_asr()
