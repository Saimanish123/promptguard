# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PromptGuard Environment Implementation.

An enterprise AI security sandbox for evaluating LLM robustness against
prompt injection attacks. The agent acts as an enterprise assistant that
must complete legitimate workplace tasks while detecting and refusing
adversarial inputs.

The environment supports three task types of increasing difficulty:
- direct-override: Obvious direct injection attempts
- indirect-rag-injection: Injections embedded inside documents
- stateful-payload-splitting: Multi-turn injections assembled across turns

Grading is fully deterministic based on tool invocation monitoring.
No NLP or LLM-as-judge components are used.
"""

import os
import sys
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Fix imports regardless of how the module is run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import PromptguardAction, PromptguardObservation
from server.scenarios import SCENARIOS, AVAILABLE_TOOLS, SYSTEM_CONTEXT_BASE
from server.grader import (
    grade_action,
    normalize_score,
    compute_episode_max,
    compute_episode_min,
)


class PromptguardEnvironment(Environment):
    """
    PromptGuard: Enterprise AI Security Sandbox.

    Evaluates LLM agent robustness against prompt injection attacks
    across three difficulty levels. Each episode presents a sequence
    of turns where the agent must invoke the correct tools on legitimate
    requests and refuse adversarial ones.

    Grader monitors tool invocations deterministically — no language
    model grading, no keyword matching as primary signal.

    Example:
        >>> env = PromptguardEnvironment()
        >>> obs = env.reset()
        >>> print(obs.task_name)  # "direct-override"
        >>> result = env.step(PromptguardAction(
        ...     tool_name="send_email",
        ...     tool_args={"to": "alice@corp.com", "subject": "Test", "body": "Hello"}
        ... ))
    """

    # Enable concurrent WebSocket sessions
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "direct-override"):
        """
        Initialize the PromptGuard environment.

        Args:
            task_name: One of direct-override, indirect-rag-injection,
                      stateful-payload-splitting
        """
        self._task_name = task_name
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Dict[str, Any]] = None
        self._current_turn: int = 0
        self._rewards: List[float] = []
        self._done: bool = False
        self._last_reward: float = 0.0
        self._conversation_history: List[Dict[str, str]] = []
        self._episode_max: float = 1.0
        self._episode_min: float = -1.0
        self._last_obs: Optional[PromptguardObservation] = None

    def _load_scenario(self) -> Dict[str, Any]:
        """Load a random scenario for the current task."""
        scenarios = SCENARIOS.get(self._task_name, [])
        if not scenarios:
            raise ValueError(f"Unknown task: {self._task_name}")
        return random.choice(scenarios)

    
    def reset(self, **kwargs):
        task_name = kwargs.get("task")
        if task_name and task_name in SCENARIOS:
            self._task_name = task_name

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = self._load_scenario()
        self._current_turn = 0
        self._rewards = []
        self._done = False
        self._last_reward = 0.0
        self._conversation_history = []

        turns = self._scenario["turns"]
        self._episode_max = compute_episode_max(turns)
        self._episode_min = compute_episode_min(turns)

        first_turn = turns[0]
        context = self._scenario.get("context", SYSTEM_CONTEXT_BASE)

        obs = PromptguardObservation(
            scenario_id=self._scenario["id"],
            task_name=self._task_name,
            turn=1,
            max_turns=len(turns),
            system_context=context + "\n\n" + SYSTEM_CONTEXT_BASE,
            user_message=first_turn["user_message"],
            conversation_history=[],
            available_tools=AVAILABLE_TOOLS,
            last_action_result=None,
            reward=0.0,
            done=False,
            metadata={"episode_id": self._state.episode_id},
        )
        self._last_obs = obs
        return obs


    def step(self, action: PromptguardAction) -> PromptguardObservation:  # type: ignore[override]
        """
        Execute one step in the environment.

        Grades the agent's tool invocation against the current turn's
        ground truth. Returns next observation with reward signal.

        Args:
            action: PromptguardAction with tool_name and tool_args

        Returns:
            PromptguardObservation with reward, done flag, and score breakdown
        """
        if self._done:
            return self._last_obs

        self._state.step_count += 1
        if self._scenario is None:
            self._scenario = self._load_scenario()
            turns = self._scenario["turns"]
            self._episode_max = compute_episode_max(turns)
            self._episode_min = compute_episode_min(turns)
        else:
            turns = self._scenario["turns"]
        turn_config = turns[self._current_turn]

        # Grade the action — fully deterministic tool call monitoring
        reward, catastrophic, info = grade_action(
            tool_name=action.tool_name,
            tool_args=action.tool_args,
            turn_config=turn_config,
        )

        self._rewards.append(reward)
        self._last_reward = reward

        # Update conversation history
        self._conversation_history.append({
            "role": "user",
            "content": turn_config["user_message"],
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": f"[Called {action.tool_name}]",
        })

        # Advance turn and check termination
        self._current_turn += 1
        episode_done = catastrophic or self._current_turn >= len(turns)
        self._done = episode_done

        # Compute normalized score
        raw_score = sum(self._rewards)
        normalized = normalize_score(
            raw_score, self._episode_min, self._episode_max
        )

        # Build next observation
        if not episode_done:
            next_turn = turns[self._current_turn]
            next_message = next_turn["user_message"]
        else:
            next_message = "[Episode complete]"

        obs = PromptguardObservation(
            scenario_id=self._scenario["id"],
            task_name=self._task_name,
            turn=self._current_turn + 1,
            max_turns=len(turns),
            system_context=(
                self._scenario.get("context", "") + "\n\n" + SYSTEM_CONTEXT_BASE
            ),
            user_message=next_message,
            conversation_history=list(self._conversation_history),
            available_tools=AVAILABLE_TOOLS,
            last_action_result=info["score_reason"],
            reward=reward,
            done=episode_done,
            metadata={
                "episode_id": self._state.episode_id,
                "score_breakdown": info,
                "normalized_score": normalized,
                "raw_score": raw_score,
                "catastrophic": catastrophic,
                "turns_completed": self._current_turn,
            },
        )
        self._last_obs = obs
        return obs

    @property
    def state(self) -> State:
        """
        Get current environment state.

        Returns last step reward (not cumulative), current observation,
        and done flag as required by OpenEnv spec.

        Returns:
            State with episode_id and step_count
        """
        return self._state