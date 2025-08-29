# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import json
import logging
from typing import Any, List, Literal, Optional

import dspy
import numpy as np

from biotopg.rag.global_query import GlobalQAProcessor
from biotopg.rag.hybrid_rag import HybridRetrieval
from biotopg.rag.local_query import LocalQAProcessor

QA_MODE = Literal["local", "global"]


class QueryManager:
    def __init__(
        self,
        retriever,
        default_mode: QA_MODE = "local",
        max_query_workers: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self.store = retriever.store
        self.logger = logger or logging.getLogger(__name__)
        self._engines: Mapping[QA_MODE, Any] = {
            "local": LocalQAProcessor(
                retriever=retriever,
                logger=self.logger,
                max_query_workers=max_query_workers,
            ),
            "global": GlobalQAProcessor(
                retriever=retriever,
                logger=self.logger,
                max_query_workers=max_query_workers,
            ),
        }
        if default_mode not in self._engines:
            raise ValueError(f"Unknown default_mode '{default_mode}'")

        self._default_mode: QA_MODE = default_mode
        self._last_mode: QA_MODE = default_mode

        self.logger.debug(
            f"QueryManager initialised with default mode {default_mode} and {max_query_workers} max query workers."
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def query(
        self,
        question,
        mode: Optional[QA_MODE] = None,
        max_iter=5,
        retriever_args: dict = {},
        *args,
        **kwargs,
    ):
        """
        Delegate to the selected QA engine.
        `mode` overrides the default just for this call.
        """
        engine_mode: QA_MODE = mode or self._default_mode
        engine = self._get_engine(engine_mode)

        self.logger.debug("Routing query to %s", engine_mode)
        answer, memory = engine.query(
            question=question,
            max_iter=max_iter,
            retriever_args=retriever_args,
            *args,
            **kwargs,
        )

        # Now we are going to process the memory to extract the resource used:
        all_fact_ids = [
            fact_id
            for mem in memory
            for walk in mem.walk_outputs
            for fact_id in walk.facts_ids
        ]
        all_fact_text = [
            fact_text
            for mem in memory
            for walk in mem.walk_outputs
            for fact_text in walk.facts
        ]
        fact_map = dict(zip(all_fact_ids, all_fact_text))

        related_passages = self.store.sqlite_db.get_passages_by_proposition_ids(
            all_fact_ids
        )
        documents = {}
        for fact_id, passage in related_passages.items():
            passage_id = passage["passage_id"]
            pmid = doc_id = passage["doc_id"]
            text = passage["page_content"]
            fact_text = fact_map.get(fact_id, "")
            if pmid not in documents:
                documents[pmid] = {
                    "passage_id": passage_id,
                    "text": text,
                    "facts": [fact_text],
                }
            else:
                documents[pmid]["facts"].append(fact_text)

        # remember where the answer came from, so show_reasoning() knows
        self._last_mode = engine_mode

        # return the answer, the fact/passages used and the memory
        return answer, documents, memory

    def show_reasoning(self, mode: Optional[QA_MODE] = None) -> None:
        """
        Print / log the reasoning steps of a particular engine.
        If `mode` is omitted, show the engine that handled the last query.
        """
        engine_mode: QA_MODE = mode or self._last_mode
        engine = self._get_engine(engine_mode)

        self.logger.debug("Showing reasoning for %s", engine_mode)
        engine.show_reasoning()

    def get_memory(
        self, mode: Optional[QA_MODE] = None, file_path: str = "memory.json"
    ):
        engine_mode: QA_MODE = mode or self._last_mode
        engine = self._get_engine(engine_mode)
        sereliazed_memory = [block.model_dump() for block in engine.memory]
        return sereliazed_memory

    def _get_engine(self, mode: QA_MODE) -> Any:
        try:
            return self._engines[mode]
        except KeyError:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes: {list(self._engines)}"
            ) from None

    def set_custom_instruction_prompt(
        self, mode: QA_MODE, predictor_name: str, instruction_prompt: str
    ):
        """
        Set a custom instruction prompt for a specific mode and predictor.

        Args:
            mode (QA_MODE): One of ['local', 'irot', 'global'].
            predictor_name (str): The name of the predictor to update (matches first part of the named predictor key).
            instruction_prompt (str): The new instruction prompt to set.

        Raises:
            ValueError: If the mode or predictor is invalid or unsupported.
        """
        if mode not in self._engines:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are {list(self._engines.keys())}"
            )

        query_engine = self._engines[mode]

        # Materialize named predictors
        matched = False
        for full_name, predictor in query_engine.named_predictors():
            base_name = full_name.split(".")[0]  # Extract base predictor name
            if base_name == predictor_name:
                if hasattr(predictor, "signature") and hasattr(
                    predictor.signature, "instructions"
                ):
                    self.logger.debug(
                        "Setting instruction prompt for '%s' in mode '%s'",
                        predictor_name,
                        mode,
                    )
                    predictor.signature.instructions = instruction_prompt
                    matched = True
                    break
                else:
                    self.logger.error(
                        f"Predictor '{predictor_name}' in mode '{mode}' does not support custom instructions."
                    )

        if not matched:
            self.logger.error(
                f"Predictor '{predictor_name}' not found in mode '{mode}'."
            )

    def set_demonstrations(
        self, mode: QA_MODE, predictor_name: str, demonstrations: List[dspy.Example]
    ):
        """
        Set demonstrations for a specific mode and predictor.

        Args:
            mode (QA_MODE): One of ['local', 'irot', 'global'].
            predictor_name (str): The name of the predictor to update (e.g., 'evaluator', 'selector', 'initial_ner').
            demonstrations (List[dspy.Example]): List of DSPy examples to use as demonstrations.

        Raises:
            ValueError: If the mode or predictor is invalid or unsupported.
        """
        if mode not in self._engines:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are {list(self._engines.keys())}"
            )

        query_engine = self._engines[mode]

        # Define the mapping of predictor names to demo attributes for each mode
        demo_mappings = {
            "local": {
                "evaluator": "demos_evaluator",
                "selector": "demos_selector",
                "last_chance": "demos_last_chance",
                "initial_ner": "demos_initial_ner",
            },
            "global": {
                "evaluator": "demos_evaluator",
                "selector": "demos_selector",
                "initial_ner": "demos_initial_ner",
            },
        }

        if predictor_name not in demo_mappings[mode]:
            available_predictors = list(demo_mappings[mode].keys())
            raise ValueError(
                f"Predictor '{predictor_name}' not available for mode '{mode}'. "
                f"Available predictors: {available_predictors}"
            )

        demo_attr_name = demo_mappings[mode][predictor_name]

        if not hasattr(query_engine, demo_attr_name):
            raise ValueError(
                f"Demo attribute '{demo_attr_name}' not found in {mode} engine"
            )

        # Validate that all items are DSPy examples
        if not all(isinstance(demo, dspy.Example) for demo in demonstrations):
            raise ValueError("All demonstrations must be dspy.Example instances")

        self.logger.debug(
            "Setting %d demonstrations for '%s' in mode '%s'",
            len(demonstrations),
            predictor_name,
            mode,
        )

        setattr(query_engine, demo_attr_name, demonstrations)

    def get_available_predictors(self, mode: QA_MODE) -> List[str]:
        """
        Get the list of available predictors for a given mode.

        Args:
            mode (QA_MODE): One of ['local', 'irot', 'global'].

        Returns:
            List[str]: List of available predictor names for the mode.

        Raises:
            ValueError: If the mode is invalid.
        """
        if mode not in self._engines:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are {list(self._engines.keys())}"
            )

        demo_mappings = {
            "local": ["evaluator", "selector", "last_chance", "initial_ner"],
            "global": ["selector", "evaluator", "initial_ner"],
        }

        return demo_mappings[mode]

    def clear_demonstrations(self, mode: QA_MODE, predictor_name: str):
        """
        Clear demonstrations for a specific mode and predictor.

        Args:
            mode (QA_MODE): One of ['local', 'irot', 'global'].
            predictor_name (str): The name of the predictor to clear demonstrations for.

        Raises:
            ValueError: If the mode or predictor is invalid or unsupported.
        """
        self.set_demonstrations(mode, predictor_name, [])
