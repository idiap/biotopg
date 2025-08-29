# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Mapping, Optional, Set, Union, cast

import dspy
import networkx as nx
import numpy as np
from tqdm import tqdm

from biotopg.rag.utils import mmr


class SelectorPrompt(dspy.Signature):
    """
    Your task is to filter a list of facts based on their relevance to a given query.
    Select only the facts that are most useful for reasoning toward the answer — they may not necessarily answer it directly.
    Each fact has an associated index. In your output, return only the indices of the selected facts — not the fact texts themselves.
    You need to be very critical and selective in your choices, as the selected facts will impact the final results.
    You must select at least 1 fact and no more than 5.
    """

    query: str = dspy.InputField(description="The input query")
    facts: str = dspy.InputField(
        description="A list of facts with their indexes to filter based on their relevance to the query."
    )
    selected_facts_indexes: List[int] = dspy.OutputField(
        description="The indexes of the selected facts."
    )


def trw_for_seeds_batch(G, seeds, num_walks, walk_length, seed_value):
    """Process multiple seeds in one worker to reduce overhead"""
    rng = np.random.default_rng(seed_value)
    all_counts = defaultdict(int)

    for seed in seeds:
        for _ in range(num_walks):
            current = seed
            all_counts[current] += 1

            for _ in range(walk_length):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = neighbors[rng.integers(0, len(neighbors))]
                all_counts[current] += 1

    return dict(all_counts)


def get_walker_ns_matrix(
    q_embedding,
    embeddings_matrix,
    q,
    damping,
    transition_symbolic,
    restart_vector,
    temperature=0.1,
    threshold=0.4,
):
    # get the symbolic_masks
    symbolic_masks = np.where(transition_symbolic > 0, 1, 0)

    # Compute the cosine similarities with the direction given by q_embedding
    cosines_t = np.dot(q_embedding, embeddings_matrix.T)

    # Apply the trheshold
    cosines_mask = cosines_t >= threshold

    # Apply the temperature scaling and the mask
    exp_cosines_t = np.exp(cosines_t / temperature)
    cosines_matrix = exp_cosines_t * symbolic_masks

    cosine_threshold_mask_matrix = cosines_mask * symbolic_masks

    # check there rows where there is no transition - because of the cosine threshold. For these rows, we will automatilly restart.
    row_sums = cosine_threshold_mask_matrix.sum(axis=1, keepdims=True)
    no_transition = np.array(row_sums.squeeze() == 0)

    neuro_matrix = cosines_matrix / cosines_matrix.sum(axis=1, keepdims=True)

    neuro_symbolic_matrix = q * transition_symbolic + (1 - q) * neuro_matrix
    neuro_symbolic_matrix[no_transition] = restart_vector

    ppr_matrix = damping * neuro_symbolic_matrix + (1 - damping) * restart_vector

    return ppr_matrix


def compute_stationary_proba(M, restart_vector, tol=1e-6, max_iter=1000):
    # Ensure restart_vector is normalized
    stationary_proba = restart_vector

    for _ in range(max_iter):
        # Update stationary probability vector
        new_stationary_proba = stationary_proba @ M

        # Check for convergence
        if np.linalg.norm(new_stationary_proba - stationary_proba, ord=1) < tol:
            return new_stationary_proba

        stationary_proba = new_stationary_proba

    # Return the final stationary probability vector
    return stationary_proba


def _process_single_query(args):
    """Process a single query - designed to be called in parallel"""
    (
        i,
        q_seed_nodes,
        q_embs,
        horizon_subgraph_nodes,
        n_nodes,
        embeddings_matrix,
        q,
        damping,
        transition_symbolic,
        temperature,
        cosine_threshold,
    ) = args

    # Create restart vector for this query
    restart_vector = np.zeros(n_nodes)
    seed_index = np.array([horizon_subgraph_nodes.index(node) for node in q_seed_nodes])

    restart_vector[seed_index] = 1.0
    restart_vector = restart_vector / np.sum(restart_vector)

    # Get the ns matrix
    M = get_walker_ns_matrix(
        q_embs,
        embeddings_matrix,
        q,
        damping,
        transition_symbolic,
        restart_vector,
        temperature=temperature,
        threshold=cosine_threshold,
    )

    stationary_proba = compute_stationary_proba(M, restart_vector)

    return i, stationary_proba


class HybridRetrieval:
    def __init__(self, store, logger=None):
        self.store = store
        self.g_full = None
        self.g_e = None
        self.g_p = None
        self.all_nodes_ids = []
        self.logger = logger or logging.getLogger(__name__)

        # For the llm feedback
        self.llm_selector = dspy.ChainOfThought(SelectorPrompt)
        self.load_graphs()

    def retrieve(
        self,
        queries: List[str],
        entities: List[List[str]],
        exclusion_list: List[List[str]],
        e_syn_k=3,
        e_syn_threshold=0.80,
        lambda_mmr=1.0,
        p_k=10,
        direct_k_multiplier=3.0,  # New parameter: multiplier for direct queries
    ):
        # how many queries to process ?
        n_queries = len(queries)
        if n_queries == 0:
            return []

        # Prepare the retrieval output structure
        retrieval_output = [
            {
                "query_embedding": np.array([]),
                "doc_text": [],
                "doc_ids": [],
                "doc_embeddings": [],
                "doc_cosines": [],
            }
            for j in range(n_queries)
        ]

        # Separate queries into entity-based and direct queries
        entity_based_indices = []
        direct_query_indices = []

        for i in range(n_queries):
            if len(entities[i]) > 0:
                entity_based_indices.append(i)
            else:
                direct_query_indices.append(i)

        if len(direct_query_indices) > 0:
            self.logger.warning(
                f"Queries at indexes {direct_query_indices} don't have associated entities, fall back to direct proposition retrieval."
            )

        # Process entity-based queries (existing logic)
        if len(entity_based_indices) > 0:
            entity_based_results = self._process_entity_based_queries(
                queries,
                entities,
                exclusion_list,
                entity_based_indices,
                e_syn_k,
                e_syn_threshold,
                lambda_mmr,
                p_k,
            )

            # Update retrieval_output with entity-based results
            for idx, result in zip(entity_based_indices, entity_based_results):
                retrieval_output[idx] = result

        # Step 3: Process direct queries
        if len(direct_query_indices) > 0:
            direct_results = self._process_direct_queries(
                queries,
                exclusion_list,
                direct_query_indices,
                lambda_mmr,
                p_k,
                direct_k_multiplier,
            )

            # Update retrieval_output with direct results
            for idx, result in zip(direct_query_indices, direct_results):
                retrieval_output[idx] = result

        return retrieval_output

    def _process_entity_based_queries(
        self,
        queries,
        entities,
        exclusion_list,
        query_indices,
        e_syn_k,
        e_syn_threshold,
        lambda_mmr,
        p_k,
    ):
        """Process queries that have entities using the existing pipeline"""

        # Extract only the queries and entities we're processing
        selected_queries = [queries[i] for i in query_indices]
        selected_entities = [entities[i] for i in query_indices]
        selected_exclusions = [exclusion_list[i] for i in query_indices]

        # Existing entity processing logic (slightly modified)
        flat_entities = [item for sublist in selected_entities for item in sublist]
        flat_entities_index = np.array(
            [
                i
                for i in range(len(selected_entities))
                for _ in range(len(selected_entities[i]))
            ]
        )

        # In total how many entities we have
        n_entities = len(flat_entities)

        # So we encode the entities
        entities_embeddings = self.store.entity_store._embedding_fn(flat_entities)

        # We retrieve the top k entities
        entities_retrieval = self.store.entity_store.query(
            query_embeddings=entities_embeddings,
            n_results=e_syn_k,
            include=["distances"],
        )

        entity_ids = entities_retrieval["ids"]
        entity_cosines = np.clip(1 - np.array(entities_retrieval["distances"]), 0, 1)

        # So we also filter the entities that do not pass a similarity threshold in the top-k
        entity_top = np.sum(entity_cosines >= e_syn_threshold, axis=1)

        # However, if nothing pass we still get the top 1
        entity_top = np.where(entity_top == 0, 1, entity_top)

        # Here we extract the ids of the selected entities
        selected_seed_entities = [
            entity_ids[i][: entity_top[i]] for i in range(n_entities)
        ]

        # Same as before, we flatten the list of selected entities before requesting the propositions
        flatten_selected_seed_entities = [
            item for sublist in selected_seed_entities for item in sublist
        ]

        # We keep track of the entity index for each selected entity to then combine the results
        entity_mask = np.array(
            [
                i
                for i in range(n_entities)
                for _ in range(len(selected_seed_entities[i]))
            ]
        )

        # Here we get the list of propositions for each selected entity
        entities2propositions = self.store.sqlite_db.get_propositions_by_entity_ids(
            flatten_selected_seed_entities
        )
        # Format
        entities2propositions = [
            entities2propositions[entity_id]
            for entity_id in flatten_selected_seed_entities
        ]

        propositions_per_queries = [[] for _ in range(len(selected_queries))]
        for idx, propositions in enumerate(entities2propositions):
            # Get the index of the query for the current entity
            q_idx = flat_entities_index[entity_mask[idx]]
            # Add the propositions to the corresponding query
            propositions_per_queries[q_idx].extend(propositions)

        # If we don't expand, we just keep the original propositions
        expanded_propositions_per_queries = propositions_per_queries

        # make a warning if id not in self.all_nodes_ids
        for i in range(len(selected_queries)):
            for _id in expanded_propositions_per_queries[i]:
                if _id not in self.all_nodes_ids:
                    self.logger.warning(
                        f"Proposition ID {_id} is not connected in the graph nodes and will not be selected."
                    )

        # For each query, we remove the exclude ids
        for i in range(len(selected_queries)):
            expanded_propositions_per_queries[i] = [
                _id
                for _id in expanded_propositions_per_queries[i]
                if _id not in selected_exclusions[i] and _id in self.all_nodes_ids
            ]

        # Continue with document store querying and MMR
        queries_to_continue = np.where(
            np.array([len(ids) > 0 for ids in expanded_propositions_per_queries])
        )[0]

        results = []

        if len(queries_to_continue) == 0:
            # Return empty results for all queries
            return [
                {
                    "query_embedding": np.array([]),
                    "doc_text": [],
                    "doc_ids": [],
                    "doc_embeddings": [],
                    "doc_cosines": [],
                }
                for _ in range(len(selected_queries))
            ]

        final_selected_queries = np.array(selected_queries)[queries_to_continue]
        selected_ids = [
            expanded_propositions_per_queries[i] for i in queries_to_continue
        ]

        # Query embeddings for selected queries
        query_embeddings = self.store.document_store._embedding_fn(
            final_selected_queries
        )

        proposition_retrieval = [
            self.store.document_store.query(
                query_embeddings=query_embeddings[i],
                include=["documents", "embeddings", "distances", "metadatas"],
                ids=selected_ids[i],
                n_results=len(selected_ids[i]),
            )
            for i in range(len(queries_to_continue))
        ]

        doc_embeddings = [
            np.array(item["embeddings"][0]) for item in proposition_retrieval
        ]
        doc_ids = [item["ids"][0] for item in proposition_retrieval]
        doc_text = [item["documents"][0] for item in proposition_retrieval]
        cosine_similarities = [
            1 - np.array(item["distances"][0]) for item in proposition_retrieval
        ]
        doc_entities_text = [
            [json.loads(item["entities_text"]) for item in q_item["metadatas"][0]]
            for q_item in proposition_retrieval
        ]
        doc_entities_ids = [
            [json.loads(item["entities_ids"]) for item in q_item["metadatas"][0]]
            for q_item in proposition_retrieval
        ]

        # Initialize results with empty dictionaries
        results = [
            {
                "query_embedding": np.array([]),
                "doc_text": [],
                "doc_ids": [],
                "doc_embeddings": [],
                "doc_cosines": [],
                "doc_entities_text": [],
                "doc_entities_ids": [],
            }
            for _ in range(len(selected_queries))
        ]

        # Apply MMR and populate results
        for j in range(len(queries_to_continue)):
            q_docs_embeddings = doc_embeddings[j]
            q_doc_cosines = cosine_similarities[j]

            # Get the top k documents based on MMR
            mmr_indices = mmr(
                q_docs_embeddings, q_doc_cosines, lambda_param=lambda_mmr, k=p_k
            )

            query_idx = queries_to_continue[j]
            results[query_idx] = {
                "query_embedding": query_embeddings[j],
                "doc_text": [doc_text[j][idx] for idx in mmr_indices],
                "doc_ids": [doc_ids[j][idx] for idx in mmr_indices],
                "doc_embeddings": [q_docs_embeddings[idx] for idx in mmr_indices],
                "doc_cosines": [q_doc_cosines[idx] for idx in mmr_indices],
                "doc_entities_text": [doc_entities_text[j][idx] for idx in mmr_indices],
                "doc_entities_ids": [doc_entities_ids[j][idx] for idx in mmr_indices],
            }

        return results

    def _process_direct_queries(
        self,
        queries,
        exclusion_list,
        query_indices,
        lambda_mmr,
        p_k,
        direct_k_multiplier,
    ):
        """Process queries without entities by directly querying the document store"""

        # Extract queries for direct processing
        direct_queries = [queries[i] for i in query_indices]
        direct_exclusions = [exclusion_list[i] for i in query_indices]

        # Calculate k for direct queries (larger to account for exclusions and MMR)
        direct_k = max(p_k, int(p_k * direct_k_multiplier))

        print(f"Using direct_k = {direct_k} for direct queries")

        # Get query embeddings
        query_embeddings = self.store.document_store._embedding_fn(direct_queries)

        results = []

        for i, (query_emb, exclusions) in enumerate(
            zip(query_embeddings, direct_exclusions)
        ):
            # Query the document store directly
            retrieval_result = self.store.document_store.query(
                query_embeddings=query_emb,
                include=["documents", "embeddings", "distances", "metadatas"],
                n_results=direct_k,
            )

            # Extract results
            doc_ids = retrieval_result["ids"][0]
            doc_text = retrieval_result["documents"][0]
            doc_embeddings = np.array(retrieval_result["embeddings"][0])
            cosine_similarities = 1 - np.array(retrieval_result["distances"][0])

            # Extract metadata if available
            try:
                doc_entities_text = [
                    json.loads(item["entities_text"]) if "entities_text" in item else []
                    for item in retrieval_result["metadatas"][0]
                ]
                doc_entities_ids = [
                    json.loads(item["entities_ids"]) if "entities_ids" in item else []
                    for item in retrieval_result["metadatas"][0]
                ]
            except (KeyError, json.JSONDecodeError):
                doc_entities_text = [[] for _ in doc_ids]
                doc_entities_ids = [[] for _ in doc_ids]

            # Filter out excluded documents and those not in graph nodes
            valid_indices = []
            filtered_doc_ids = []
            filtered_doc_text = []
            filtered_doc_embeddings = []
            filtered_cosines = []
            filtered_entities_text = []
            filtered_entities_ids = []

            for idx, doc_id in enumerate(doc_ids):
                if doc_id not in exclusions and doc_id in self.all_nodes_ids:
                    valid_indices.append(idx)
                    filtered_doc_ids.append(doc_id)
                    filtered_doc_text.append(doc_text[idx])
                    filtered_doc_embeddings.append(doc_embeddings[idx])
                    filtered_cosines.append(cosine_similarities[idx])
                    filtered_entities_text.append(doc_entities_text[idx])
                    filtered_entities_ids.append(doc_entities_ids[idx])

            if len(filtered_doc_embeddings) == 0:
                # No valid documents found
                result = {
                    "query_embedding": query_emb,
                    "doc_text": [],
                    "doc_ids": [],
                    "doc_embeddings": [],
                    "doc_cosines": [],
                    "doc_entities_text": [],
                    "doc_entities_ids": [],
                }
            else:
                # Apply MMR to filtered results
                filtered_doc_embeddings = np.array(filtered_doc_embeddings)
                filtered_cosines = np.array(filtered_cosines)

                mmr_indices = mmr(
                    filtered_doc_embeddings,
                    filtered_cosines,
                    lambda_param=lambda_mmr,
                    k=min(p_k, len(filtered_doc_embeddings)),
                )

                result = {
                    "query_embedding": query_emb,
                    "doc_text": [filtered_doc_text[idx] for idx in mmr_indices],
                    "doc_ids": [filtered_doc_ids[idx] for idx in mmr_indices],
                    "doc_embeddings": [
                        filtered_doc_embeddings[idx] for idx in mmr_indices
                    ],
                    "doc_cosines": [filtered_cosines[idx] for idx in mmr_indices],
                    "doc_entities_text": [
                        filtered_entities_text[idx] for idx in mmr_indices
                    ],
                    "doc_entities_ids": [
                        filtered_entities_ids[idx] for idx in mmr_indices
                    ],
                }

            results.append(result)

        return results

    def load_graphs(self, force_reload=False, masks_ontology_ids: List[str] = []):
        if self.g_full is None or force_reload:
            self.logger.debug("Loading the graph ...")
            # Load the graph from the SQLite database
            self.g_e, self.g_p, self.g_full = self.store.sqlite_db.get_graphs(
                masks_ontology_ids=masks_ontology_ids
            )
            self.all_nodes_ids = list(self.g_full.nodes)
            self.logger.debug("Graph loaded successfully.")

    def show_graph_statistics(self):
        n_nodes = self.g_full.number_of_nodes()
        n_edges = self.g_full.number_of_edges()
        self.logger.info(f"Full Graph has {n_nodes} nodes and {n_edges} edges.")

    def get_horizon(
        self,
        initial_seed_nodes,
        use_passage_links=True,
        num_walks=500,
        walk_length=3,
        top_k=1000,
        seed=42,
        batch_size=None,
    ):
        self.load_graphs()
        self.logger.debug(f"Getting the graph using passage links: {use_passage_links}")

        max_workers = min(len(initial_seed_nodes), os.cpu_count() or 1)

        # Do we use the links between passages ?
        if use_passage_links:
            working_graph = self.g_full
        else:
            # Use only the entity graph
            working_graph = self.g_e

        # Determine batch size - larger batches for fewer worker spawns
        if batch_size is None:
            batch_size = max(10, len(initial_seed_nodes) // (max_workers or 4))

        # Create seed batches
        seed_batches = [
            initial_seed_nodes[i : i + batch_size]
            for i in range(0, len(initial_seed_nodes), batch_size)
        ]

        node_counts = Counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    trw_for_seeds_batch,
                    working_graph,
                    batch,
                    num_walks,
                    walk_length,
                    seed,
                )
                for batch in seed_batches
            ]

            for future in futures:
                counts = future.result()
                node_counts.update(counts)

        # Convert to dict for consistency
        node_counts = dict(node_counts)

        if top_k is not None:
            top_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            horizon_nodes = set(node for node, _ in top_nodes)
        else:
            horizon_nodes = set(node_counts.keys())

        return working_graph.subgraph(horizon_nodes).copy()

    def _run_walkers(
        self,
        query_embeddings,
        seed_nodes: List[List[str]],
        q=0.5,
        damping=0.5,
        horizon_threshold=1e-4,
        temperature=0.1,
        cosine_threshold=0.4,
        max_workers=None,  # New parameter for controlling parallelism
        use_passage_links=True,  # Use passage links by default
    ):
        assert query_embeddings.shape[0] == len(seed_nodes), (
            "Number of queries must match number of seed nodes"
        )

        _seed_nodes = list(set([sn for l_sn in seed_nodes for sn in l_sn]))

        # Get the horizon subgraph
        start_time = time.time()
        horizon_subgraph = self.get_horizon(
            initial_seed_nodes=_seed_nodes,
            top_k=1000,
            use_passage_links=use_passage_links,
        )
        end_time = time.time()
        self.logger.debug(
            f"Horizon subgraph created in {end_time - start_time:.2f} seconds."
        )

        # Get the adjacency matrix
        am = nx.adjacency_matrix(horizon_subgraph, weight="weight").todense()

        # Get the transition matrix (symbolic)
        transition_symbolic = am / am.sum(axis=1, keepdims=True)

        # Ensure the symbolic transition matrix and embeddings match in terms of nodes
        horizon_subgraph_nodes = list(horizon_subgraph.nodes)

        # Extract the text of the associated proposition nodes
        _horizon_subgraph_nodes = self.store.document_store.get(
            ids=horizon_subgraph_nodes, include=["documents", "embeddings"]
        )
        horizon_subgraph_nodes_texts = dict(
            zip(
                _horizon_subgraph_nodes["ids"],
                _horizon_subgraph_nodes["documents"],
            )
        )

        # Reorder the embeddings
        indexes = np.array(
            [
                _horizon_subgraph_nodes["ids"].index(node)
                for node in horizon_subgraph_nodes
            ]
        )
        embeddings_matrix = np.array(_horizon_subgraph_nodes["embeddings"][indexes])

        # Prepare data for parallel processing
        n_nodes = horizon_subgraph.number_of_nodes()
        n_queries = query_embeddings.shape[0]

        # measure the time to run
        start_time = time.time()

        # Create arguments for each query
        query_args = []
        for i in range(n_queries):
            args = (
                i,  # query index
                seed_nodes[i],  # q_seed_nodes
                query_embeddings[i],  # q_embs
                horizon_subgraph_nodes,  # horizon_subgraph_nodes
                n_nodes,  # n_nodes
                embeddings_matrix,  # embeddings_matrix
                q,  # q
                damping,  # damping
                transition_symbolic,  # transition_symbolic
                temperature,  # temperature
                cosine_threshold,  # cosine_threshold
            )
            query_args.append(args)

        # Process queries in parallel
        all_stationary_probas = [None] * n_queries  # Pre-allocate to maintain order

        # Determine number of workers (default to CPU count)
        if max_workers is None:
            max_workers = min(n_queries, os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(_process_single_query, args): args[0]
                for args in query_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    query_index, stationary_proba = future.result()
                    all_stationary_probas[query_index] = stationary_proba
                except Exception as exc:
                    query_index = future_to_index[future]
                    print(f"Query {query_index} generated an exception: {exc}")
                    raise exc

        all_stationary_probas = np.array(all_stationary_probas)

        # final time
        end_time = time.time()
        self.logger.debug(
            f"Walkers run completed in {end_time - start_time:.2f} seconds with {max_workers} workers."
        )

        return (
            all_stationary_probas,
            horizon_subgraph_nodes,
            horizon_subgraph_nodes_texts,
            embeddings_matrix,
        )

    def run_walkers(
        self,
        query_embeddings,
        seed_nodes,
        groups=None,  # New parameter: numpy array like [0,0,0,1,1,2,2]
        exclusion_list=[],
        top_k=20,
        q=0.5,
        damping=0.85,
        horizon_threshold=1e-4,
        temperature=0.1,
        cosine_threshold=0.4,
        use_passage_links=True,
        **kwargs,
    ):
        # Get the walker results
        (
            all_stationary_probas,
            horizon_subgraph_nodes,
            horizon_subgraph_nodes_texts,
            embeddings_matrix,
        ) = self._run_walkers(
            query_embeddings=query_embeddings,
            seed_nodes=seed_nodes,
            q=q,
            damping=damping,
            horizon_threshold=horizon_threshold,
            temperature=temperature,
            cosine_threshold=cosine_threshold,
            use_passage_links=use_passage_links,
        )

        # If no groups specified, treat all queries as one group (backward compatibility)
        if groups is None:
            groups = np.zeros(len(seed_nodes), dtype=int)

        # Validate groups array
        assert len(groups) == len(seed_nodes), (
            "Groups array length must match number of queries"
        )
        assert len(groups) == query_embeddings.shape[0], (
            "Groups array length must match query embeddings"
        )

        # Get unique groups and sort them to maintain order
        unique_groups = np.unique(groups)

        # Prepare results list - one result dict per group
        results_by_group = []

        # Process each group independently
        for group_id in unique_groups:
            # Find all query indices belonging to this group
            group_mask = groups == group_id
            group_indices = np.where(group_mask)[0]

            # Extract stationary probabilities for this group only
            group_stationary_probas = all_stationary_probas[group_indices]

            # Compute cumulative walkers for this group (sum across group queries)
            group_cumulative_walkers = group_stationary_probas.sum(axis=0)

            # Compute attribution for this group (argmax across group queries)
            group_attributed_walkers = np.argmax(group_stationary_probas, axis=0)

            # Filter out nodes in the exclusion list
            filtered_indexes = [
                idx
                for idx in range(len(horizon_subgraph_nodes))
                if horizon_subgraph_nodes[idx] not in exclusion_list
            ]

            # Handle empty case
            if len(filtered_indexes) == 0:
                group_result = {
                    "doc_text": [],
                    "doc_ids": [],
                    "p_walkers": [],
                    "attributed_walkers": [],
                    "doc_embeddings": [],
                }
                results_by_group.append(group_result)
                continue

            # Filter cumulative walkers and attributed walkers based on filtered indexes
            filtered_group_cumulative_walkers = group_cumulative_walkers[
                filtered_indexes
            ]
            filtered_group_attributed_walkers = group_attributed_walkers[
                filtered_indexes
            ]

            # Extract the top-k indexes from the filtered cumulative walkers for this group
            top_k_indexes = np.argsort(filtered_group_cumulative_walkers)[::-1][:top_k]

            # Map top-k indexes to nodes and their corresponding data
            top_k_nodes = [
                horizon_subgraph_nodes[filtered_indexes[idx]] for idx in top_k_indexes
            ]
            top_k_nodes_texts = [
                horizon_subgraph_nodes_texts[node] for node in top_k_nodes
            ]
            top_k_walkers = filtered_group_cumulative_walkers[top_k_indexes]
            top_k_node_embeddings = np.array(
                embeddings_matrix[filtered_indexes][top_k_indexes]
            )
            top_k_attributed_walkers = filtered_group_attributed_walkers[top_k_indexes]

            # Prepare the result for this group
            group_result = {
                "doc_text": top_k_nodes_texts,
                "doc_ids": top_k_nodes,
                "p_walkers": top_k_walkers,
                "attributed_walkers": top_k_attributed_walkers,
                "doc_embeddings": top_k_node_embeddings,
            }

            results_by_group.append(group_result)

        # Return list of results, one per group, in group order
        return results_by_group
