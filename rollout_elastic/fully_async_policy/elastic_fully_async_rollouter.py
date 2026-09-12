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
"""FT-aware ``FullyAsyncRollouter`` defined as a real subclass.

``FullyAsyncRollouter`` is ``@ray.remote(num_cpus=10, max_concurrency=100)``-
decorated. A decorator patch applied in the driver (or in any process) on the
``ActorClass`` is not guaranteed to reach the worker that instantiates the actor
— Ray re-imports the class from its defining module there, so runtime patches
are lost. To make the fault-tolerance extension survive cross-process
deserialization, every FT method lives in this subclass body, and the subclass
is re-decorated below with the same resource spec as the parent.

With fault tolerance disabled this class is behaviourally identical to the
native ``FullyAsyncRollouter``.
"""

from __future__ import annotations

import asyncio
import logging
import os

import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import safe_create_task
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter as _RemoteFullyAsyncRollouter,
)
from verl.workers.rollout.llm_server import LLMServerManager

# ``FullyAsyncRollouter`` is `@ray.remote(num_cpus=10, max_concurrency=100)`-decorated.
# Unwrap to subclass; re-decorate at the bottom with the same spec.
_BaseFullyAsyncRollouter = _RemoteFullyAsyncRollouter.__ray_actor_class__


class ElasticFullyAsyncRollouterImpl(_BaseFullyAsyncRollouter):
    """FullyAsyncRollouter + FT Supervisor / token continuation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fault tolerance is constructed before the trainer's first sync and
        # started by init_ft_supervisor so CKE-first failures are reportable.
        self._ft_supervisor = None
        self._trainer_handle = None
        logging.getLogger(__name__).warning(
            "[FT] ElasticFullyAsyncRollouter.__init__: pid=%d cls=%s.%s ft.enabled=%s",
            os.getpid(),
            type(self).__module__,
            type(self).__qualname__,
            OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=None),
        )

    def get_load_balancer(self):
        """Get the global load balancer for FT-enabled checkpoint manager construction."""
        return self.llm_server_manager.global_load_balancer

    async def init_ft_supervisor(self, trainer_handle):
        """Construct the ThreadedSupervisor for inference instance elasticity.

        Called after the trainer's CKE is set up (via set_rollouter), so the
        cross-actor callbacks can safely reach the trainer's checkpoint_manager.

        The Supervisor lives in the Rollouter because it owns the replicas +
        LB + spawn_replacement. CKE membership notifications cross the actor
        boundary via Ray RPC to the trainer.

        Args:
            trainer_handle: Ray actor handle for FullyAsyncTrainer, used for
                cross-actor CKE membership notifications (on_replica_dead /
                on_replica_added).
        """
        import logging as _ft_logging

        self._trainer_handle = trainer_handle
        _ft_log = _ft_logging.getLogger(__name__)

        ft_cfg = None
        try:
            from verl.workers.rollout.fault_tolerance import FaultToleranceConfig

            ft_node = OmegaConf.select(self.config, "async_training.fault_tolerance")
            if ft_node is not None:
                ft_cfg = FaultToleranceConfig(**OmegaConf.to_container(ft_node, resolve=True))
        except Exception:
            ft_cfg = None

        self._ft_supervisor = None
        if ft_cfg is None or not ft_cfg.enabled:
            _ft_log.warning("[FT] init_ft_supervisor: fault_tolerance not enabled, skipping Supervisor")
            return

        from verl.workers.rollout.fault_tolerance import (
            Supervisor,
            ThreadedSupervisor,
            make_on_dead,
        )

        async def probe_fn(replica):
            try:
                return bool(await asyncio.wait_for(replica.health(), timeout=2.0))
            except Exception:
                return False

        replicas = self.llm_server_manager.get_replicas()
        # replica_id = _server_address (matches LB's server_id keying)
        replica_map = {r._server_address: r for r in replicas}

        # Cross-actor callback: notify Trainer so its CKE prunes the dead
        # replica. The Manager owns all communication-group reset decisions.
        async def ckpt_mgr_callback(replica_id):
            if self._trainer_handle is None:
                return
            try:
                ref = self._trainer_handle._on_replica_dead_from_supervisor.remote(replica_id)
                await asyncio.wrap_future(ref.future())
            except Exception as e:
                _ft_log.warning(
                    "[FT] ckpt_mgr_callback: failed to notify trainer of replica death %s: %s",
                    replica_id,
                    e,
                )

        spawner_fn = None
        on_spawn_success_fn = None
        if getattr(ft_cfg, "replace_dead_replicas", False):

            async def spawner_fn(dead_id):  # noqa: E306
                return await self.llm_server_manager.spawn_replacement(dead_id)

            async def on_spawn_success_fn(dead_id, new_replica):  # noqa: E306
                # Register as pending first; Manager promotes it after a full
                # sync at the current target version, then LB admission is safe.
                if self._trainer_handle is None:
                    raise RuntimeError("trainer handle is unavailable while registering a replacement replica")
                ref = self._trainer_handle._on_replica_added_from_supervisor.remote(new_replica)
                await asyncio.wrap_future(ref.future())
                sup = getattr(self, "_ft_supervisor", None)
                if sup is not None:
                    sup.supervisor.add_replica(new_replica._server_address, new_replica)
                _ft_log.warning(
                    "[FT] on_spawn_success: replica %s added back",
                    new_replica._server_address,
                )

        on_dead_handler = make_on_dead(
            lb_handle=self.llm_server_manager.global_load_balancer,
            replica_to_server_ids=lambda rid: [rid],
            ckpt_mgr_callback=ckpt_mgr_callback,
            spawner=spawner_fn,
            on_spawn_success=on_spawn_success_fn,
        )

        async def promote_fn(servers):
            await self.llm_server_manager.global_load_balancer.add_servers.remote(servers)

        inner_sup = Supervisor(
            replicas=replica_map,
            probe_fn=probe_fn,
            on_dead=on_dead_handler,
            promote_fn=promote_fn,
            interval_s=ft_cfg.heartbeat_interval_s,
            miss_threshold=ft_cfg.heartbeat_miss_threshold,
            probe_timeout_s=2.0,
        )
        # Own thread+loop so rollouter's blocking operations can't starve heartbeat.
        self._ft_supervisor = ThreadedSupervisor(inner_sup)
        # The fully-async main performs its initial parameter sync immediately
        # after this method returns, so heartbeat/reporting must already run.
        self._ft_supervisor.start()
        _ft_log.warning(
            "[FT] init_ft_supervisor: ThreadedSupervisor started with %d replicas, interval=%s miss_threshold=%s",
            len(replica_map),
            ft_cfg.heartbeat_interval_s,
            ft_cfg.heartbeat_miss_threshold,
        )

    def report_sync_failure(self, replica_id: str, source: str = "unknown") -> None:
        """Forward a CKE sync failure to the rollouter-owned Supervisor."""
        supervisor = getattr(self, "_ft_supervisor", None)
        if supervisor is None:
            return
        supervisor.report_failure(replica_id, source)

    async def promote_synced_replica(
        self,
        replica_id: str,
        servers: dict,
        attempt_id: int,
        target_version: int,
    ) -> bool:
        """Serialize serving admission with Supervisor death handling."""
        supervisor = getattr(self, "_ft_supervisor", None)
        if supervisor is None:
            return False
        return await supervisor.promote_replica(
            replica_id,
            servers,
            attempt_id,
            target_version,
        )

    async def _init_async_rollout_manager(self):
        import logging as _ft_logging

        _ft_logging.getLogger(__name__).warning(
            "[FT] ElasticFullyAsyncRollouter._init_async_rollout_manager entered: pid=%d cls=%s",
            os.getpid(),
            type(self).__name__,
        )
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        self.async_rollout_mode = True
        self.llm_server_manager = await LLMServerManager.create(config=self.config)
        await self._init_fully_async_progress()
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(fully_async=True),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=self.teacher_model_manager.get_client() if self.teacher_model_manager else None,
        )

    async def _init_fully_async_progress(self):
        """Mode C: create + initialise the RolloutProgressStoreActor if enabled.

        Mirrors one_step_off_policy's _init_one_step_progress. Only runs when both
        ``async_training.fault_tolerance.enabled`` and
        ``async_training.fault_tolerance.progress.enabled`` are True. The store handle
        is wired into every FullyLLMServerClient produced by
        ``llm_server_manager.get_client(fully_async=True)``, which is what turns on the
        token-continuation retry path.
        """
        import logging as _ft_logging

        _log = _ft_logging.getLogger(__name__)
        try:
            ft_enabled = bool(OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False))
            progress_enabled = bool(
                OmegaConf.select(self.config, "async_training.fault_tolerance.progress.enabled", default=False)
            )
        except Exception as e:
            _log.exception("[FT] init fully-async progress failed: %s", e)
            return
        if not ft_enabled or not progress_enabled:
            _log.warning(
                "[FT] fully-async token continuation skipped (ft.enabled=%s, progress.enabled=%s)",
                ft_enabled,
                progress_enabled,
            )
            return

        try:
            progress_node = OmegaConf.select(self.config, "async_training.fault_tolerance.progress")
            progress_config = self._build_progress_config(progress_node)
        except Exception:
            _log.exception("[FT] _build_progress_config failed — Mode C will NOT be enabled")
            raise
        try:
            await self.llm_server_manager._init_progress_store(progress_config)
        except Exception:
            _log.exception("[FT] _init_progress_store failed — Mode C will NOT be enabled")
            raise
        _log.warning(
            "[FT] fully-async Mode C (token continuation) enabled: run_id=%s, persist_root=%s",
            self.llm_server_manager.run_id,
            progress_config.persist_root,
        )

    def _build_progress_config(self, progress_node):
        """Map the config ``progress`` node onto a ProgressConfig dataclass."""
        from verl.workers.rollout.fault_tolerance import ModelVersionPolicy, ProgressConfig

        if progress_node is None:
            return ProgressConfig()
        kwargs = {}
        for key, value in OmegaConf.to_container(progress_node, resolve=True).items():
            if key == "model_version_policy" and isinstance(value, dict):
                kwargs[key] = ModelVersionPolicy(mode=value.get("mode", "exact"))
            else:
                kwargs[key] = value
        return ProgressConfig(**kwargs)

    async def fit(self):
        """Start the async rollouter — FT-aware Supervisor lifecycle."""

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # The Supervisor normally started in init_ft_supervisor, before the
        # trainer's initial sync. Keep a guarded fallback for direct callers.
        if getattr(self, "_ft_supervisor", None) is not None:
            import logging as _ft_logging

            if not self._ft_supervisor.is_running:
                _ft_logging.getLogger(__name__).warning("[FT] fit: starting Supervisor heartbeat")
                self._ft_supervisor.start()
            else:
                _ft_logging.getLogger(__name__).debug("[FT] fit: Supervisor heartbeat already running")
        else:
            import logging as _ft_logging

            _ft_logging.getLogger(__name__).warning("[FT] fit: _ft_supervisor is None — no FT detection")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True
            self._resume_event.set()

        # Create the main asynchronous task
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

            # FT: stop Supervisor heartbeat
            if getattr(self, "_ft_supervisor", None) is not None:
                self._ft_supervisor.stop()

        print("[FullyAsyncRollouter] Rollouter fit completed")


# Re-decorate with the same resource spec as the parent.
ElasticFullyAsyncRollouter = ray.remote(num_cpus=10, max_concurrency=100)(ElasticFullyAsyncRollouterImpl)
