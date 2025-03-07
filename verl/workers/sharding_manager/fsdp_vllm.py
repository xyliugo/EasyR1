# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.model_utils import print_gpu_memory_usage
from ..rollout.vllm_rollout import load_dtensor_weights
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh = None,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        print_gpu_memory_usage("Before state_dict() in sharding manager")
        actor_weights = self.module.state_dict()
        print_gpu_memory_usage("After state_dict() in sharding manager")

        self.inference_engine.wake_up()
        load_dtensor_weights(
            actor_weights, self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        )
        print_gpu_memory_usage("After sync model weights in sharding manager")

        del actor_weights
        torch.cuda.empty_cache()
        print_gpu_memory_usage("After del state_dict and empty_cache in sharding manager")
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        print_gpu_memory_usage("Before vllm offload in sharding manager")
        self.inference_engine.sleep(level=1)
        print_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        torch.cuda.empty_cache()  # add empty cache after each compute

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        group = vllm_ps.get_tensor_model_parallel_group().device_group

        prev_device = data.batch.device
        data.batch = data.batch.cuda(device=torch.cuda.current_device())
        data.batch = VF.allgather_dict_tensors(data.batch.contiguous(), size=tp_size, group=group, dim=0)
        data.batch = data.batch.to(prev_device)
        # all gather non_tensor_batch
        all_non_tensor_batch = [None for _ in range(tp_size)]
        torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
        data.non_tensor_batch = {
            k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch
        }
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        dp_rank = dist.get_rank()
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            data = data.chunk(chunks=tp_size)[dp_rank % tp_size]

        return data
