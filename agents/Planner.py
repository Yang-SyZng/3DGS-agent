from __future__ import annotations

import logging

from prompts import PlannerDescription, PlannerPrompt

from .Baser import BaseFunctionAgent

logger = logging.getLogger(__name__)


class Planner(BaseFunctionAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "Planner")
        kwargs.setdefault("description", PlannerDescription)
        kwargs.setdefault("system_prompt", PlannerPrompt)
        super().__init__(*args, **kwargs)
