# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Literal, Optional

import dspy
import numpy as np
from colorama import Fore, Style, init
from pydantic import BaseModel

from biotopg.rag.hybrid_rag import HybridRetrieval


class LocalWalkOutput(BaseModel):
    question: str
    facts: List[str]
    facts_ids: List[str]


class LocalMemoryBlock(BaseModel):
    reasoning: str
    is_sufficient: Optional[bool] = None
    walk_outputs: List[LocalWalkOutput]
    next_questions: Optional[List[str]] = None


class LocalPromptSelector(dspy.Signature):
    """
    You are a retriever agent that has access to a bank of facts.
    You are given a question and a list of retrieved facts formatted as an indexed list.
    Your task is to select facts as relevant facts to help the user answer the question.

    Select facts that make progress toward answering the question, even if they don't provide the complete answer. Look for facts that:
    - Answer the question directly (best case)
    - Answer part of the question or a sub-question
    - Provide relevant context or background information needed for the answer
    - Connect to entities, concepts, or relationships mentioned in the question

    Return the list of the indexes (numbers) of all the selected facts.

    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="The indexed retrieved facts")
    selected_facts: List[int] = dspy.OutputField(
        description="The selected facts indexes"
    )


class LocalPromptEvaluator(dspy.Signature):
    """
    # TASK
    You are an agent specialized in complex question answering and reasoning.
    You will be given a question and a set of collected facts (potentially empty).

    # FACTS
    All the relevant facts that have been collected so far.

    # GUIDELINES
    By combining the information from the collected facts, determine if you can answer the question.

    - If YES, return is_sufficient = True and answer the question.
    - If NO, then it means you need more information from the fact bank to answer the question. Return is_sufficient = False and plan the `next questions` for collecting more facts.

    When planning `next_questions`,
    1) Identify what is missing — what do you still need to know - considering the information from the already collected facts ?
    2) What are the most relevant direction to explore ?

    Reason strategically step by step.

    When proposing a question:
        - `entity` refer to the named entity or object onto which the question will apply.
        - `question` formulates the request.
    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="Some collected relevant facts")
    is_sufficient: bool = dspy.OutputField(
        description="Whether the facts are sufficient to answer the question"
    )
    answer: str = dspy.OutputField(description="The answer (if sufficient)")
    next_questions: List[str] = dspy.OutputField(
        description="Next retrieval question (if necessary)"
    )


class LastAnswerPrompt(dspy.Signature):
    """
    # TASK
    You are an agent specialized in complex question answering and reasoning.
    You will be given a question and a set of collected facts.

    # Facts
    All the relevant facts that have been collected.

    The collected facts may be incomplete.

    It is your last chance to answer the question, so be critical and try to provide the best answer possible.
    Only respond with the potential answer.
    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="Collected relevant facts")
    answer: str = dspy.OutputField(description="The answer")


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


class LocalQAProcessor(dspy.Module):
    def __init__(self, retriever: Any, logger=None, max_query_workers=1, **kwargs):
        self.retriever = retriever
        self.evaluator = dspy.ChainOfThought(LocalPromptEvaluator)
        self.initial_ner = dspy.Predict(PromptNER)
        self.selector = dspy.ChainOfThought(LocalPromptSelector)
        self.last_chance = dspy.ChainOfThought(LastAnswerPrompt)
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
        self.demos_last_chance = []
        self.demos_initial_ner = []

    def reset_memory(self):
        """
        Reset the memory.
        """
        self.memory = []
        self.collected_facts_ids = set()

    def get_facts_memory(self) -> str:
        facts = []
        for block in self.memory:
            for walk_output in block.walk_outputs:
                for fact in walk_output.facts:
                    if fact not in facts:
                        facts.append(fact)
        # Join all parts with newlines
        return facts

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

        # Get the query embeddings and the initial facts
        query_embeddings = np.array([initial_retrievals[0]["query_embedding"]])

        initial_seed_nodes = initial_retrievals[0]["doc_ids"]
        initial_seed_facts = initial_retrievals[0]["doc_text"]
        formatted_facts = "\n".join(
            [f"{idx}: {fact}" for idx, fact in enumerate(initial_seed_facts)]
        )

        seed_indexes = self.selector(
            question=question, facts=formatted_facts
        ).selected_facts

        # Well, if nothing have been selected, we will still try to answer with all we can found, maybe we will be lucky !
        if not seed_indexes:
            self.logger.warning(
                "No precise indexes selected. Using all as a start by default then."
            )
            seed_indexes = list(range(len(initial_seed_nodes)))

        # We filter the seed nodes and facts based on the selected indexes
        seed_nodes = [initial_seed_nodes[i] for i in seed_indexes]
        seed_nodes_text = [initial_seed_facts[i] for i in seed_indexes]

        return seed_nodes, seed_nodes_text, query_embeddings

    def _process_group(self, question, group_retrieval):
        retrieved_facts = group_retrieval["doc_text"]
        retrieved_facts_ids = group_retrieval["doc_ids"]
        formatted_facts = "\n".join(
            [f"{idx}: {fact}" for idx, fact in enumerate(retrieved_facts)]
        )
        selected_facts_indexes = self.selector(
            question=question, facts=formatted_facts
        ).selected_facts

        # Check that the selected facts are valid indexes
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

        selected_facts_ids = [retrieved_facts_ids[j] for j in selected_facts_indexes]
        selected_facts_text = [retrieved_facts[j] for j in selected_facts_indexes]
        return selected_facts_ids, selected_facts_text

    def query(
        self,
        question: str,
        max_iter: int = 2,
        retriever_args: dict = {},
        *args,
        **kwargs,
    ):
        """
        Given a question, return the answer.
        """
        # reset the memory
        self.reset_memory()

        original_question = question

        is_sufficient = False
        iteration = 0
        answer = ""

        if not retriever_args:
            retriever_args = self._default_retriever_args.copy()

        self.logger.debug(f"Retriever args: {retriever_args}")

        entities = self.initial_ner(
            question=question, demos=self.demos_initial_ner
        ).entities

        self.logger.debug(f"Extracted entities: {entities}")

        seed_nodes, seed_nodes_text, query_embeddings = self.initial_seeding(
            question=question, entities=entities, retriever_args=retriever_args
        )
        n = len(seed_nodes)
        self.logger.debug(f"{n} seed nodes : {seed_nodes_text}")

        # return seed_nodes, query_embeddings

        # For the first iteration, there can be only one group and one question
        groups = np.array([0])
        n_groups = 1
        questions = [question]

        # The seed nodes are going to be used as the initial facts for the first iteration - they can be already added in the memory.
        walk_block = LocalWalkOutput(
            question=question,
            facts=seed_nodes_text,
            facts_ids=seed_nodes,
        )
        self.memory.append(
            LocalMemoryBlock(
                reasoning="This is the initial retrieval seeding step.",
                is_sufficient=False,
                walk_outputs=[walk_block],
                next_questions=None,
            )
        )

        # Prepare seed nodes:
        seed_nodes = [seed_nodes]

        while not is_sufficient and iteration < max_iter:
            self.logger.debug(
                "Iteration %d: Processing questions and groups ...", iteration + 1
            )

            # run the walkers per group of questions
            global_retrieval_output = self.retriever.run_walkers(
                query_embeddings=query_embeddings,
                seed_nodes=seed_nodes,
                exclusion_list=list(self.collected_facts_ids),
                max_workers=None,
                groups=groups,
                **retriever_args,
            )

            assert len(global_retrieval_output) == n_groups, (
                "Output size of the retriever should match the number of groups."
            )

            all_selected_facts_ids = [None] * n_groups
            all_selected_facts_text = [None] * n_groups
            unique_new_collected_facts = set()
            unique_new_collected_facts_ids = set()

            # We call the Selector(s) agent
            selector_start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.max_query_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_group, questions[i], group_retrieval
                    ): i
                    for i, group_retrieval in enumerate(global_retrieval_output)
                }

                for future in as_completed(futures):
                    i = futures[future]
                    selected_facts_ids, selected_facts_text = future.result()
                    all_selected_facts_ids[i] = selected_facts_ids
                    all_selected_facts_text[i] = selected_facts_text

                    self.logger.debug(
                        f"Group {i}: Question {questions[i]}\nSelected facts IDs: {selected_facts_ids}\nText: {selected_facts_text}"
                    )

            # Prepare the evaluator input
            selector_end_time = time.time()
            elapsed_selector_time = selector_end_time - selector_start_time
            self.logger.debug(
                f"Selector took {elapsed_selector_time:.2f} seconds to process {len(groups)} group(s)."
            )
            # Update the collected_facts_ids:
            for group_selected_facts_ids in all_selected_facts_ids:
                self.collected_facts_ids.update(group_selected_facts_ids)
                unique_new_collected_facts_ids.update(group_selected_facts_ids)

            for selected_facts_text in all_selected_facts_text:
                unique_new_collected_facts.update(selected_facts_text)

            # Combined the memory and the new collected facts
            formated_new_facts = list(unique_new_collected_facts)
            memory_facts = self.get_facts_memory()
            facts_for_evaluation = []
            for fact in memory_facts + formated_new_facts:
                if fact not in facts_for_evaluation:
                    facts_for_evaluation.append(fact)
            formated_facts_for_evaluation = "\n".join(facts_for_evaluation)

            self.logger.debug(
                f"Formatted facts for evaluation:\n{formated_facts_for_evaluation}"
            )

            # Evaluation
            self.logger.debug("Calling the evaluator agent ...")
            evaluator_response = self.evaluator(
                question=original_question,
                facts=formated_facts_for_evaluation,
                demos=self.demos_evaluator,
            )

            is_sufficient = evaluator_response.is_sufficient
            reasoning = evaluator_response.reasoning

            self.logger.debug(
                f"Evaluator response: is_sufficient={is_sufficient}, answer={evaluator_response.answer}, reasoning={reasoning}"
            )

            # Update memory
            self.logger.debug("Updating memory with the current iteration results ...")
            walks_groups_blocks = [
                LocalWalkOutput(
                    question=questions[i],
                    facts=all_selected_facts_text[i],
                    facts_ids=all_selected_facts_ids[i],
                )
                for i in range(len(groups))
            ]

            # If not we try to continue with the next questions
            questions = evaluator_response.next_questions

            self.memory.append(
                LocalMemoryBlock(
                    reasoning=reasoning,
                    is_sufficient=is_sufficient,
                    walk_outputs=walks_groups_blocks,
                    next_questions=questions if questions else None,
                )
            )

            # now is it sufficient ?
            if is_sufficient:
                answer = evaluator_response.answer
                return answer, self.memory

            iteration += 1
            n_groups = len(questions)
            groups = np.array(range(n_groups))

            self.logger.debug(
                f"Iteration {iteration}: New questions for the next iteration: {questions}"
            )

            # If nothing at all had bee selected, we use the ones of the previous step.
            if len(unique_new_collected_facts_ids) == 0:
                self.logger.warning(
                    "No new facts were selected, using the previously collected facts."
                )
                unique_new_collected_facts_ids = self.collected_facts_ids
                if not unique_new_collected_facts_ids:
                    self.logger.warning(
                        "No collected facts found. Returning empty answer."
                    )
                    return "I don't know.", self.memory

            # We also start from the newly selected facts as seed nodes for the next iteration
            seed_nodes = [list(unique_new_collected_facts_ids)] * n_groups
            query_embeddings = np.array(
                self.retriever.store.document_store._embedding_fn(questions)
            )

        # If after max_iter iterations we still don't have a sufficient answer,
        # we give it a last chance to answer the question
        self.logger.warning(
            "We reach the maximum number of iteration, trying to answer the question still ..."
        )
        memory_facts = self.get_facts_memory()
        last_chance_response = self.last_chance(
            question=question,
            facts="\n".join(memory_facts),
            demos=self.demos_last_chance,
        )
        return last_chance_response.answer, self.memory

    def show_reasoning(self) -> None:
        """
        Outputs the step-by-step reasoning from the memory blocks directly to the console with colors.
        """
        for iteration, block in enumerate(self.memory):
            # Step header in bold and cyan
            step_header = f"{Style.BRIGHT}{Fore.CYAN}Step {iteration + 1}:"
            print(step_header)

            # Reasoning in yellow
            reasoning = f"{Fore.YELLOW}Reasoning: {block.reasoning}"
            print(reasoning)

            # Questions in green
            if block.walk_outputs:
                for walk_output in block.walk_outputs:
                    question = f"{Fore.GREEN}  - Question: {walk_output.question} ("
                    facts = f"    {Fore.MAGENTA}Facts: {', '.join(walk_output.facts) if walk_output.facts else 'No facts selected'}"
                    print(question)
                    print(facts)
            else:
                print(f"{Fore.RED}No question in this step.")

            # Add a blank line for separation between steps
            print()
