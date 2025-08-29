# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import json
import os
from typing import Any, List, Mapping

import dspy
from dotenv import load_dotenv

load_dotenv()


def load_and_configure_llm(llm_config: Mapping[str, Any]):
    """
    Loads and configures a dspy LLM with JSONAdapter.
    Uses local LLM if api_base is provided, else uses external API.
    """

    api_base = llm_config.get("api_base", None)
    llm_name = llm_config.get("llm_name", "gpt-3.5-turbo")

    # Check if there is an api_key in the config or environment variables
    api_key = llm_config.get("api_key", os.getenv("OPENAI_API_KEY"))
    max_tokens = llm_config.get("max_tokens", 4096)

    if api_base:
        lm = dspy.LM(llm_name, api_base=api_base, max_tokens=max_tokens)
    elif api_key is not None:
        lm = dspy.LM(llm_name, api_key=api_key, max_tokens=max_tokens)
    else:
        raise ValueError(
            "Either api_base or api_key must be provided in the configuration."
        )

    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    return lm


import dspy


def get_cost(lm):
    """Get the cost of the model."""
    cost = sum([x["cost"] for x in lm.history if x["cost"] is not None])
    return cost
