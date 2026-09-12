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
"""FT-aware ``FullyAsyncTrainer`` defined as a real subclass.

``FullyAsyncTrainer`` is ``@ray.remote(num_cpus=10)``-decorated. A decorator
patch applied on the ``ActorClass`` is not guaranteed to reach the worker that
instantiates the actor — Ray re-imports the class from its defining module
there, so runtime patches are lost. To make the fault-tolerance extension
survive cross-process deserialization, every FT method lives in this subclass
body, and the subclass is re-decorated below with the same resource spec as
the parent.

With fault tolerance disabled this class is behaviourally identical to the
native ``FullyAsyncTrainer``.
"""

from __future__ import annotations

import asyncio

import ray
from omegaconf import OmegaConf

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer as _RemoteFullyAsyncTrainer,
)
from verl.utils.config import omega_conf_to_dataclass

# ``FullyAsyncTrainer`` is `@ray.remote(num_cpus=10)`-decorated.
# Unwrap to subclass; re-decorate at the bottom with the same spec.
_BaseFullyAsyncTrainer = _RemoteFullyAsyncTrainer.__ray_actor_class__


class ElasticFullyAsyncTrainerImpl(_BaseFullyAsyncTrainer):
    """FullyAsyncTrainer + FT checkpoint manager wiring + cross-actor callbacks."""

    def _setup_checkpoint_manager(self, rollouter):
        """Setup checkpoint manager after rollouter is initialized (FT-aware)."""
        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)

        # FT: read fault_tolerance config + fetch LB handle from rollouter
        ft_cfg = None
        try:
            from verl.workers.rollout.fault_tolerance import FaultToleranceConfig

            ft_node = OmegaConf.select(self.config, "async_training.fault_tolerance")
            if ft_node is not None:
                ft_cfg = FaultToleranceConfig(**OmegaConf.to_container(ft_node, resolve=True))
        except Exception:
            ft_cfg = None

        lb_handle = None
        try:
            lb_handle = ray.get(rollouter.get_load_balancer.remote())
        except Exception:
            lb_handle = None

        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_wg,
            replicas=replicas,
            fault_tolerance=ft_cfg,
            load_balancer_handle=lb_handle,
            sync_failure_reporter=(lambda replica_id, source: rollouter.report_sync_failure.remote(replica_id, source)),
            replica_promotion_reporter=(
                lambda replica_id, servers, attempt_id, target_version: (
                    rollouter.promote_synced_replica.remote(
                        replica_id,
                        servers,
                        attempt_id,
                        target_version,
                    )
                )
            ),
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _on_replica_dead_from_supervisor(self, replica_id: str):
        """Cross-actor callback: Rollouter's Supervisor detected a dead replica.

        Prunes the replica from the trainer-side CKE and marks membership dirty
        so the next update_weights rebuilds the NCCL group without it.
        """
        if self.checkpoint_manager is not None:
            await self.checkpoint_manager.on_replica_dead(replica_id)

    async def _on_replica_added_from_supervisor(self, new_replica):
        """Cross-actor callback: Rollouter's Supervisor spawned a replacement.

        Register the replacement as pending.  The next complete weight-sync
        transaction promotes it and only then admits it to the load balancer.
        """
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.add_pending_replicas([new_replica])


# Re-decorate with the same resource spec as the parent.
ElasticFullyAsyncTrainer = ray.remote(num_cpus=10)(ElasticFullyAsyncTrainerImpl)
