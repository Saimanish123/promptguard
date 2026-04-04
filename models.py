# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PromptGuard Environment.

PromptGuard is an enterprise AI security sandbox where an LLM agent must
complete legitimate workplace tasks while detecting and refusing prompt
injection attacks. The agent interacts via structured tool calls, and
the grader monitors exactly which tools were invoked.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PromptguardAction(Action):
    """
    Action for the PromptGuard environment.

    The agent selects a tool to invoke and provides the required arguments.
    Available tools:
    - send_email(to, subject, body): Send an email
    - execute_query(sql): Execute a database query
    - approve_candidate(candidate_id): Approve a job candidate
    - summarize_content(text): Summarize provided text
    - refuse_request(reason): Refuse and log a malicious or unsafe request
    """

    tool_name: str = Field(
        ...,
        description=(
            "Name of the tool to invoke. Must be one of: "
            "send_email, execute_query, approve_candidate, "
            "summarize_content, refuse_request"
        ),
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool as a key-value dictionary",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning for the tool choice (not graded)",
    )


class PromptguardObservation(Observation):
    """
    Observation from the PromptGuard environment.

    Each observation represents one turn in the enterprise assistant scenario.
    The agent must read the user_message, consider the conversation_history,
    and decide which tool to invoke.
    """

    scenario_id: str = Field(
        default="",
        description="Unique identifier for the current scenario",
    )
    task_name: str = Field(
        default="",
        description="Current task name: direct-override, indirect-rag-injection, or stateful-payload-splitting",
    )
    turn: int = Field(
        default=0,
        description="Current turn number within the episode",
    )
    max_turns: int = Field(
        default=5,
        description="Maximum number of turns in this episode",
    )
    system_context: str = Field(
        default="",
        description="Role description and available tools for the agent",
    )
    user_message: str = Field(
        default="",
        description="The current user input the agent must respond to",
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous turns in the conversation as list of role/content dicts",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names the agent can invoke",
    )
    last_action_result: Optional[str] = Field(
        default=None,
        description="Result of the previous tool invocation, if any",
    )
    reward: float = Field(
        default=0.0,
        description="Reward received for the last action",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional grading info: score breakdown per dimension",
    )
