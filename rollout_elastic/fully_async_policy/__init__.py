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
"""FT-aware subclasses of the fully-async policy actors.

``FullyAsyncRollouter`` and ``FullyAsyncTrainer`` are ``@ray.remote`` classes;
decorator patches on the ``ActorClass`` do not survive cross-process actor
instantiation, so the fault-tolerance extension is provided as real subclasses
(``ElasticFullyAsyncRollouter`` / ``ElasticFullyAsyncTrainer``) whose methods
live in the class body. ``rollout_elastic.patch.experimental`` swaps these
classes in when it wires ``FullyAsyncTaskRunner``.
"""

from .elastic_fully_async_rollouter import ElasticFullyAsyncRollouter
from .elastic_fully_async_trainer import ElasticFullyAsyncTrainer

__all__ = ["ElasticFullyAsyncRollouter", "ElasticFullyAsyncTrainer"]
