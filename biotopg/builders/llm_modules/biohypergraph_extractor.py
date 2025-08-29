# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import hashlib
import json
import logging
import random
from typing import List

import dspy
import numpy as np
from dateutil.parser import parse
from langchain_core.documents.base import Document
from pydantic import BaseModel
from tqdm import tqdm

from biotopg.utils.biomodels import (
    BioEntity,
    BioHyperProposition,
    BioHyperPropositionList,
    HyperProposition,
)


def is_date(string):
    try:
        parse(string, fuzzy=False)
        return True
    except ValueError:
        return False


class PromptHyperPropositionizerWithNER(dspy.Signature):
    """
    Your task is to extract all meaningful decontextualized propositions from the given medical passage, in order to connect the provided Biomedical Named Entities.
    - Propositions connect biomedical entities: they describe their relations, links.
    - Propositions represent distinct and minimal pieces of meaning.
    - Propositions are decontextualised, meaning they are independently fully interpretable without the context of the initial passage.
    - Ensure full coverage so that, together, the hyperpropositions capture every explicit biomedical connections/relationships mentioned in the passage.

    Rules for each proposition:
    1. Complete & Standalone: a proposition convey exactly one fact or relationship, including all necessary context.
    2. Entity Usage: Use only the entities from Named Entities; do not introduce any others.
    . Full Coverage: Collect propositions that together capture every meaningful point in the Passage.
    . Preserve context: add all necessary contextual elements to improve clarity and precision of each propositions without the context of the initial passage.
    . Focus of factual statements and conclusive results as stated in the passage.
    . Avoid extracting propositions that refer to the methodology, experimental setup, or other non-factual or conclusive elements unless they directly contribute to the understanding of the relations between biomedical entities.

    The precision and completeness of these propositions directly impact the performance and reliability of downstream tasks, particularly the quality of question answering systems. Therefore, careful adherence to the above rules is critical to ensure factual accuracy, unambiguous interpretation, and full semantic coverage.

    You will be given some examples.
    """

    passage = dspy.InputField(description="The input passage")
    entities: List[str] = dspy.InputField(
        description="A list of biomedical entities to use for the extraction"
    )
    hyperpropositions: List[HyperProposition] = dspy.OutputField(
        description="A JSON list of propositions and their associated entities"
    )


class BioHyperPropositionizerWithNER(dspy.Module):
    def __init__(
        self,
        logger: logging.Logger = None,
        extractor_demonstration_file: str = None,
        include_supplementary_entities: bool = False,
    ):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.hyperpropositionizer = dspy.Predict(PromptHyperPropositionizerWithNER)
        self.extractor_demonstrations = self.load_demonstrations(
            extractor_demonstration_file
        )
        self.logger.info(
            f"Loaded {len(self.extractor_demonstrations)} demonstrations for the extractor."
        )

    def set_custom_instruction_prompt(
        self, predictor_name: str, instruction_prompt: str
    ):
        """
        Set a custom instruction prompt for a specific mode and predictor.
        """

        # Materialize named predictors
        matched = False
        for full_name, predictor in self.named_predictors():
            base_name = full_name.split(".")[0]  # Extract base predictor name
            self.logger.debug(
                f"Checking predictor: {base_name} against {predictor_name}"
            )
            if base_name == predictor_name:
                if hasattr(predictor, "signature") and hasattr(
                    predictor.signature, "instructions"
                ):
                    self.logger.info(
                        f"Setting instruction prompt for {predictor_name} in mode",
                    )
                    predictor.signature.instructions = instruction_prompt
                    matched = True
                    break
                else:
                    self.logger.warning(
                        f"Predictor '{predictor_name}' in mode does not support custom instructions."
                    )

        if not matched:
            self.logger.error(f"Predictor '{predictor_name}' not found.")

    def load_demonstrations(self, demonstration_file: str) -> List[dspy.Example]:
        """Load examples from a JSON file."""
        if not demonstration_file:
            return []

        with open(demonstration_file, "r") as f:
            examples = json.load(f)
        demonstrations = [
            dspy.Example(**example).with_inputs("passage", "entities")
            for example in examples
        ]
        return demonstrations

    def extract_propositions(
        self,
        passage: Document,
        retries: int = 3,
    ):
        passage_content = passage.page_content

        # Extract the NER entities from the PubTator annotation
        bio_entities = passage.metadata["entities"]
        all_bio_entities = list(
            dict.fromkeys(
                entity for entities in bio_entities.values() for entity in entities
            )
        )
        map_entities_to_ids = {
            entity_text.lower(): entity_id
            for entity_id, entity_texts in bio_entities.items()
            for entity_text in entity_texts
        }

        # Always be sure that first attemp is done with temperature 0.0
        dspy.settings.lm.kwargs["temperature"] = 0.0

        # Retry until success or max attempts
        success = False
        for attempt in range(retries):
            if attempt > 0:
                # It is the first retry, we gonna try increasing the temperature by sampling a temperature between 0.0 and 1.0
                temperature = random.uniform(0.0, 1.0)
                dspy.settings.lm.kwargs["temperature"] = temperature

            # call the LLM
            try:
                llm_response = self.hyperpropositionizer(
                    passage=passage_content,
                    entities=all_bio_entities,
                    demos=self.extractor_demonstrations,
                )

                hyperproposition_list = llm_response.hyperpropositions
                success = True
            except ValidationError as e:
                self.logger.error(f"Validation error for passage {passage_id}: {e}")
                success = False
            except TimeoutError as e:
                self.logger.error(f"Timeout error for passage {passage_id}: {e}")
                success = False
            except Exception as e:
                self.logger.error(f"Unexpected error for passage {passage_id}: {e}")
                success = False

        # If all attempts failed, return an empty list
        if success:
            # Map entities to their IDs and create BioHyperProposition objects
            biohyperpropositions = []
            for hyperprop in hyperproposition_list:
                mapped_entities = [
                    BioEntity(
                        id=map_entities_to_ids.get(
                            entity.lower(),
                            f"Unknown|{hashlib.sha256(entity.encode('utf-8')).hexdigest()}",
                        ),
                        text=entity,
                    )
                    for entity in hyperprop.entities
                    if not entity.isdigit() and not is_date(entity)
                ]
                biohyperpropositions.append(
                    BioHyperProposition(
                        proposition=hyperprop.proposition,
                        entities=mapped_entities,
                    )
                )

            # Create the BioHyperPropositionList object
            biohyperproposition_list = BioHyperPropositionList(
                biohyperpropositions=biohyperpropositions
            )
            return biohyperproposition_list
        else:
            self.logger.warning(
                "All retry attempts failed. Returning an empty PropositionList."
            )
            return BioHyperPropositionList(**{"biohyperpropositions": []})
