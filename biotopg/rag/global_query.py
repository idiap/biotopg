# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Literal, Optional

import dspy
import numpy as np
from colorama import Fore, Style, init
from pydantic import BaseModel

from biotopg.rag.hybrid_rag import HybridRetrieval


class GlobalWalkOutput(BaseModel):
    seed_nodes_ids: List[str] = []
    facts: List[str]
    facts_ids: List[str] = []


class GlobalMemoryBlock(BaseModel):
    reasoning: str
    updated_answer: str
    walk_outputs: List[GlobalWalkOutput]


class GlobalPromptSelector(dspy.Signature):
    """
    You are a retriever agent with access to a bank of facts.

    You are given:
    1. A broad or high-level question.
    2. An existing abstract answer (can be empty on the first iteration).
    3. A list of **new indexed facts**, which may:
    • Reinforce or expand the answer
    • Introduce ambiguity or contradictions
    • Be irrelevant or off-topic

    Task:
    Select the indexes of *all* new facts you judge relevant enough to update or enrich the answer.

    Guidelines:
    - Prefer facts that meaningfully expand, complete, or nuance the answer.
    - Ignore facts that are irrelevant, redundant, or drift too far from the original question’s scope.
    """

    question: str = dspy.InputField(description="The original abstract question")
    current_answer: str = dspy.InputField(
        description="The existing abstract answer to refine"
    )
    facts: str = dspy.InputField(description="Indexed list of new facts")
    selected_facts: List[int] = dspy.OutputField(
        description="Indexes of the new facts used in the update"
    )


class GlobalPromptEvaluator(dspy.Signature):
    """
    # Task Overview
    You are assisting in refining a **conceptual or thematic answer** to an abstract question. You are provided with:

    1. A broad or high-level question.
    2. An existing abstract answer to that question (can be empty if this is the first iteration).
    3. A list of new relevant facts.

    Your responsibility is to produce an **updated answer** that integrates the new information from the provided facts to create a more comprehensive answer.

    # Guidelines
    - Generate a refined or expanded answer by integrating the information from the new relevant facts.
    - Do not make anything up. Do not include information not provided by the facts.
    - The new answer should **retain and build upon** the information from the previous answer, rather than replacing it.
    - Make the answer **well organized and structured**:
    - Use markdown formatting (e.g., headings, bullet points, numbered lists) where appropriate.
    - Ensure it is clear and avoid repetition.
    - Present ideas logically and coherently.
    - Maintain a professional and thoughtful tone.
    - If helpful, include a brief concluding sentence that summarizes the main insight.
    """

    question: str = dspy.InputField(description="The original abstract question")
    previous_answer: str = dspy.InputField(
        description="The existing abstract answer to refine"
    )
    facts: str = dspy.InputField(description="Indexed list of new facts")

    updated_answer: str = dspy.OutputField(
        description="The updated abstract answer, refined or expanded based on new facts"
    )


class PromptNER(dspy.Signature):
    """
    # TASK
    You are an entity extraction agent. Your task is to extract named entities from the given input question.

    # GUIDELINES
    - Identify all named entities in the question.
    - Return the extracted entities as a list of strings. There can be one or multiple entities.
    """

    question: str = dspy.InputField(description="The input question")
    entities: List[str] = dspy.OutputField(description="The extracted named entities")


def split_indexes_uniformly(n, m):
    if m > n:
        print("Number of groups cannot exceed the number of indexes.")
        m = n
    # Shuffle the indexes randomly
    indexes = list(range(n))
    random.shuffle(indexes)

    # Calculate the size of each group
    n = len(indexes)
    group_size = n // m
    remainder = n % m  # Extra elements to distribute

    # Create groups
    _groups_ = []
    start = 0
    for i in range(m):
        end = start + group_size + (1 if i < remainder else 0)
        _groups_.append(indexes[start:end])
        start = end

    groups = np.array([0] * n)
    for i, group in enumerate(_groups_):
        for idx in group:
            groups[idx] = i

    return groups


class GlobalQAProcessor(dspy.Module):
    def __init__(
        self,
        retriever: HybridRetrieval,
        logger=None,
        max_query_workers=1,
        **kwargs,
    ):
        self.retriever = retriever
        self.initial_ner = dspy.Predict(PromptNER)
        self.selector = dspy.ChainOfThought(GlobalPromptSelector)
        self.evaluator = dspy.ChainOfThought(GlobalPromptEvaluator)
        self.memory = []
        self.logger = logger or logging.getLogger(__name__)
        self.collected_facts_ids = set()
        self.max_query_workers = max_query_workers
        self._default_retriever_args = {
            "initial_retriever_args": {
                "e_syn_k": 3,
                "e_syn_threshold": 0.80,
                "lambda_mmr": 1.0,
                "p_k": 10,
            },
            "q": 0.3,
            "damping": 0.85,
            "cosine_threshold": 0.4,
            "horizon_threshold": 1e-4,
            "temperature": 0.1,
            "top_k": 15,
        }

        self.demos_evaluator = []
        self.demos_selector = []
        self.demos_initial_ner = []

    def reset_memory(self):
        """
        Reset the memory.
        """
        self.memory = []
        self.collected_facts_ids = set()

    def initial_seeding(self, question: str, entities: List[str], retriever_args: dict):
        initial_retrievals = self.retriever.retrieve(
            queries=[question],
            entities=[entities],
            exclusion_list=[[]],
            **retriever_args["initial_retriever_args"],
        )

        if not initial_retrievals:
            self.logger.warning("No initial retrievals found. Returning empty answer.")
            return []

        initial_seed_nodes = initial_retrievals[0]["doc_ids"]
        initial_seed_facts = initial_retrievals[0]["doc_text"]
        formatted_facts = "\n".join(
            [f"{idx}: {fact}" for idx, fact in enumerate(initial_seed_facts)]
        )
        attributed_walker = np.array([0] * len(initial_seed_nodes))
        initial_retrievals[0]["attributed_walkers"] = attributed_walker

        seed_indexes = self.selector(
            question=question, current_answer="", facts=formatted_facts
        ).selected_facts

        return seed_indexes, initial_retrievals[0]

    def process_query_iteration(
        self,
        group_selected_indexes,
        grouped_retrieval,
        query_embeddings,
        groups,
        m=1,
        alpha=1.0,
        beta=0.7,
        gamma=0.15,
    ):
        n_groups = len(np.unique(groups))
        assert len(group_selected_indexes) == n_groups, (
            "Number of groups does not match the number of seed indexes."
        )
        assert len(grouped_retrieval) == n_groups, (
            "Number of groups does not match the number of retrievals."
        )
        assert query_embeddings.shape[0] == len(groups), (
            "Number of query embeddings does not match the number of groups."
        )

        retrieval_doc_map = {}

        for k, g_retrieval in enumerate(grouped_retrieval):
            g_selected_idx = group_selected_indexes[k]

            # First, get the negative (not selected facts) per group
            n = len(g_retrieval["doc_ids"])
            g_excluded_facts_indexes = [
                idx for idx in range(n) if idx not in g_selected_idx
            ]
            g_excluded_facts_embeddings = np.array(
                [g_retrieval["doc_embeddings"][idx] for idx in g_excluded_facts_indexes]
            )
            if len(g_excluded_facts_embeddings) == 0:
                g_excluded_facts_embeddings = np.zeros_like(
                    g_retrieval["doc_embeddings"][0]
                )
            else:
                g_excluded_facts_embeddings = np.mean(
                    g_excluded_facts_embeddings, axis=0
                )

            # Now for the selected ids
            for idx in g_selected_idx:
                doc_id = g_retrieval["doc_ids"][idx]
                if not doc_id in retrieval_doc_map:
                    retrieval_doc_map[doc_id] = {}
                    retrieval_doc_map[doc_id]["doc_text"] = g_retrieval["doc_text"][idx]
                    retrieval_doc_map[doc_id]["doc_embeddings"] = g_retrieval[
                        "doc_embeddings"
                    ][idx]

                # Get the corresponding query embedding
                g_attributed_walker = g_retrieval["attributed_walkers"][idx]
                attributed_walker = np.where(groups == k)[0][g_attributed_walker]
                q_emb = query_embeddings[attributed_walker]
                if "query_embedding" not in retrieval_doc_map[doc_id]:
                    retrieval_doc_map[doc_id]["query_embedding"] = []
                    retrieval_doc_map[doc_id]["query_embedding"].append(q_emb)
                else:
                    retrieval_doc_map[doc_id]["query_embedding"].append(q_emb)

                # Fill with the excluded facts embeddings
                if "excluded_facts_embeddings" not in retrieval_doc_map[doc_id]:
                    retrieval_doc_map[doc_id]["excluded_facts_embeddings"] = []
                    retrieval_doc_map[doc_id]["excluded_facts_embeddings"].append(
                        g_excluded_facts_embeddings
                    )
                else:
                    retrieval_doc_map[doc_id]["excluded_facts_embeddings"].append(
                        g_excluded_facts_embeddings
                    )

        # Now we merge back to get the matrix for query updates
        all_new_seed_nodes = list(retrieval_doc_map.keys())

        # Update the list of collected facts ids
        self.collected_facts_ids.update(all_new_seed_nodes)

        n_seeds = len(all_new_seed_nodes)
        _query_embeddings = np.array(
            [
                np.array(retrieval_doc_map[doc_id]["query_embedding"]).mean(axis=0)
                for doc_id in all_new_seed_nodes
            ]
        )
        selected_doc_embeddings = np.array(
            [
                retrieval_doc_map[doc_id]["doc_embeddings"]
                for doc_id in all_new_seed_nodes
            ]
        )
        excluded_facts_embeddings = np.array(
            [
                np.array(retrieval_doc_map[doc_id]["excluded_facts_embeddings"]).mean(
                    axis=0
                )
                for doc_id in all_new_seed_nodes
            ]
        )

        query_embeddings = (
            alpha * _query_embeddings
            + beta * selected_doc_embeddings
            - gamma * excluded_facts_embeddings
        )

        groups = split_indexes_uniformly(n_seeds, m)
        seed_nodes = [[all_new_seed_nodes[i]] for i in range(len(all_new_seed_nodes))]

        return query_embeddings, seed_nodes, groups

    def _process_group(self, question, current_answer, group_retrieval):
        retrieved_facts = group_retrieval["doc_text"]
        formatted_facts = "\n".join(
            [f"{idx}: {fact}" for idx, fact in enumerate(retrieved_facts)]
        )
        selected_facts_indexes = self.selector(
            question=question, current_answer=current_answer, facts=formatted_facts
        ).selected_facts

        # Check that they are valid indexes
        n_facts = len(retrieved_facts)
        if any(idx < 0 or idx >= n_facts for idx in selected_facts_indexes):
            self.logger.warning(
                f"Invalid selected indexes in: {selected_facts_indexes}"
            )
            # can we see waht happened ?
            dspy.inspect_history(1)
            selected_facts_indexes = [
                idx for idx in selected_facts_indexes if 0 <= idx < n_facts
            ]

        return selected_facts_indexes

    def query(
        self,
        question: str,
        max_iter: int = 5,
        retriever_args: dict = {},
        m=1,
        alpha=1.0,
        beta=0.7,
        gamma=0.15,
        *args,
        **kwargs,
    ):
        answer = ""
        self.reset_memory()

        if not retriever_args:
            retriever_args = self._default_retriever_args.copy()

        self.logger.debug(f"Retriever args: {retriever_args}")

        entities = self.initial_ner(
            question=question, demos=self.demos_initial_ner
        ).entities

        self.logger.debug(f"Extracted entities: {entities}")

        seed_indexes, initial_retrieval = self.initial_seeding(
            question=question, entities=entities, retriever_args=retriever_args
        )

        if not seed_indexes:
            self.logger.warning(
                "No seed nodes found. Let's try to still continue will all of them ..."
            )
            seed_indexes = list(range(len(initial_retrieval["doc_text"])))

        n = len(seed_indexes)
        self.logger.debug(f"Number of seed nodes: {n}")

        query_embeddings = np.array([initial_retrieval["query_embedding"]] * n)
        groups = np.array([0] * n)

        # In the first iteration we also directly include the seed facts.
        initial_facts_text = [
            initial_retrieval["doc_text"][idx] for idx in seed_indexes
        ]
        self.logger.debug(
            "Initial facts collected: {}".format(", ".join(initial_facts_text))
        )
        initial_facts_ids = [initial_retrieval["doc_ids"][idx] for idx in seed_indexes]
        unique_new_collected_facts = initial_facts_text

        # prepare the first iteration
        query_embeddings, seed_nodes, groups = self.process_query_iteration(
            group_selected_indexes=[seed_indexes],
            grouped_retrieval=[initial_retrieval],
            query_embeddings=query_embeddings,
            groups=groups,
            m=m,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        # Put the initial facts in the memory
        walks_groups_blocks = [
            GlobalWalkOutput(
                seed_nodes_ids=[],
                facts=initial_facts_text,
                facts_ids=initial_facts_ids,
            )
        ]

        self.memory.append(
            GlobalMemoryBlock(
                reasoning="This is the initial retrieval.",
                updated_answer="",
                walk_outputs=walks_groups_blocks,
            )
        )

        iteration = 0
        while iteration < max_iter:
            self.logger.info("Starting iteration {}...".format(iteration + 1))

            global_retrieval_output = self.retriever.run_walkers(
                query_embeddings=query_embeddings,
                seed_nodes=seed_nodes,
                exclusion_list=list(self.collected_facts_ids),
                max_workers=None,
                groups=groups,
                **retriever_args,
            )

            # start calling the selected
            selector_start_time = time.time()

            n_groups = len(np.unique(groups))
            self.logger.debug(
                "There are {} groups of walkers to process.".format(n_groups)
            )

            # Process each group in parallel
            all_selected_facts_indexes = [None] * n_groups
            all_selected_facts_ids = [None] * n_groups
            all_selected_facts_text = [None] * n_groups

            with ThreadPoolExecutor(max_workers=self.max_query_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_group, question, answer, group_retrieval
                    ): i
                    for i, group_retrieval in enumerate(global_retrieval_output)
                }

                for future in as_completed(futures):
                    i = futures[future]
                    selected_facts_text_indexes = future.result()

                    self.logger.debug(
                        "Group {} selected {} facts.".format(
                            i, len(selected_facts_text_indexes)
                        )
                    )

                    # Get the checked and selected indexes
                    all_selected_facts_indexes[i] = selected_facts_text_indexes

                    # Get the associated facts and ids
                    all_selected_facts_ids[i] = [
                        global_retrieval_output[i]["doc_ids"][j]
                        for j in selected_facts_text_indexes
                    ]
                    all_selected_facts_text[i] = [
                        global_retrieval_output[i]["doc_text"][j]
                        for j in selected_facts_text_indexes
                    ]

            # Prepare the evaluator input
            selector_end_time = time.time()
            elapsed_selector_time = selector_end_time - selector_start_time
            self.logger.debug(
                f"Selector took {elapsed_selector_time:.2f} seconds to process {len(groups)} group(s)."
            )

            # Collecte all the selected facts from all groups in order
            for group_selected_facts_text in all_selected_facts_text:
                for fact in group_selected_facts_text:
                    if fact not in unique_new_collected_facts:
                        # Avoid duplicates
                        unique_new_collected_facts.append(fact)

            self.logger.debug(
                "All new collected facts: {}".format(
                    ", ".join(unique_new_collected_facts)
                )
            )

            # Combined the memory and the new collected facts
            formated_new_facts = "\n".join(unique_new_collected_facts)

            # Evaluation
            self.logger.debug("Calling the evaluator with the question ...")
            evaluator_response = self.evaluator(
                question=question,
                previous_answer=answer,
                facts=formated_new_facts,
                demos=[],
            )
            self.logger.debug("Evaluator updated the answer !")

            # reset after evaluator
            unique_new_collected_facts = []

            # Update memory
            self.logger.debug("Updating the memory with the new facts ...")
            walks_groups_blocks = [
                GlobalWalkOutput(
                    seed_nodes_ids=[seed_nodes[j][0] for j in np.where(groups == i)[0]],
                    facts=all_selected_facts_text[i],
                    facts_ids=all_selected_facts_ids[i],
                )
                for i in range(n_groups)
            ]

            self.memory.append(
                GlobalMemoryBlock(
                    reasoning=evaluator_response.reasoning,
                    updated_answer=evaluator_response.updated_answer,
                    walk_outputs=walks_groups_blocks,
                )
            )
            # Get the updated answer
            answer = evaluator_response.updated_answer
            # We check if there are any new facts selected for the next iteration
            if all(
                len(g_selected_facts_indexes) == 0
                for g_selected_facts_indexes in all_selected_facts_indexes
            ):
                self.logger.info("No new facts selected. Stopping the query.")
                break

            self.logger.debug("Preparing for the next iteration ...")
            query_embeddings, seed_nodes, groups = self.process_query_iteration(
                group_selected_indexes=all_selected_facts_indexes,
                grouped_retrieval=global_retrieval_output,
                query_embeddings=query_embeddings,
                groups=groups,
                m=m,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            iteration += 1

        return answer, self.memory

    def show_reasoning(self) -> None:
        """
        Outputs the step-by-step reasoning from the memory blocks directly to the console with colors.
        """
        for iteration, block in enumerate(self.memory):
            # Step header in bold and cyan
            step_header = f"{Style.BRIGHT}{Fore.CYAN}Step {iteration + 1}:"
            print(step_header)

            # Questions in green
            for j, walk_output in enumerate(block.walk_outputs):
                facts = f"{Fore.MAGENTA}TEAM {j + 1}\nFacts: {', '.join(walk_output.facts) if walk_output.facts else 'No facts selected'}"
                print(facts)

            updated_answer = f"{Fore.BLUE}Updated Answer: {block.updated_answer}"
            print(updated_answer)

            # Reasoning in yellow
            reasoning = f"{Fore.YELLOW}Reasoning: {block.reasoning}"
            print(reasoning)

            # Add a blank line for separation between steps
            print()
