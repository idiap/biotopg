# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import glob
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from logging import DEBUG, INFO
from typing import Any, List, Mapping, cast

import chromadb
import numpy as np
from chromadb import Collection, Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from dateutil.parser import parse
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer

from biotopg.builders.sql_db import MapDB
from biotopg.utils.logger import get_std_logger


def from_dict_to_adjacency_matrix(mapping, row_to_index, col_to_index):
    adj_matrix = np.zeros((len(row_to_index), len(col_to_index)), dtype=int)

    # Step 4: Populate the adjacency matrix
    for prop, entities in mapping.items():
        row_idx = row_to_index[prop]
        for entity in entities:
            col_idx = col_to_index[entity]
            adj_matrix[row_idx, col_idx] = 1

    return adj_matrix


class BioHybridBaseLoader:
    def __init__(
        self,
        storage,
        config: Mapping[str, Any] = {},
        logger=None,
    ):
        self.storage = storage
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def generate_id(self, entity: str) -> str:
        return hashlib.sha256(entity.encode("utf-8")).hexdigest()

    def index_passages(self, passages: List[Document]) -> None:
        self.storage.sqlite_db.insert_passages(
            passages, self.storage.document_store.name()
        )

    def index_propositions(
        self,
        input_propositions: List[Document],
    ) -> None:
        self.logger.info("Loading propositions.")
        propositions = deepcopy(
            input_propositions
        )  # important to deepcopy. But we could fix this later

        n = len(propositions)
        loading_batch_size = self.config.get("chroma_loading_batch_size", 512)
        for i in range(0, n, loading_batch_size):
            index_start = i
            index_end = min(i + loading_batch_size, n)
            self.logger.info(f"Loading propositions from {index_start} to {index_end}.")

            # Create a batch of documents
            batch_propositions = propositions[index_start:index_end]

            # extract the ids
            batch_proposition_ids = [
                proposition.metadata["id"] for proposition in batch_propositions
            ]

            # Extract the propositions text
            self.logger.debug(
                "Calling embeddings function on the propositions.",
            )
            batch_propositions_texts = [
                proposition.page_content for proposition in batch_propositions
            ]
            batch_propositions_passages_ids = [
                proposition.metadata["passage_id"] for proposition in batch_propositions
            ]

            # Extract the entities
            batch_dict_of_entities = defaultdict(set)

            all_propositions_entities_ontology_ids = [
                proposition.metadata.get("entities_ids")
                for proposition in batch_propositions
            ]
            all_propositions_entities_texts = [
                proposition.metadata.get("entities_text")
                for proposition in batch_propositions
            ]
            # Here, we collect all the propositions in the current batch that use this entity. We build as a set of ensure it is unique during insert.
            for i, entities in enumerate(all_propositions_entities_ontology_ids):
                for entity in entities:
                    batch_dict_of_entities[entity].add(i)

            # Convert back to list
            for key in batch_dict_of_entities:
                batch_dict_of_entities[key] = list(batch_dict_of_entities[key])

            # Here, we create a distinct list of tuple for all entity ids and their text utterances
            all_distinct_pairs = list(
                set(
                    [
                        (entity_id, entity_text)
                        for entities_ids, entities_texts in zip(
                            all_propositions_entities_ontology_ids,
                            all_propositions_entities_texts,
                        )
                        for entity_id, entity_text in zip(
                            entities_ids or [],
                            entities_texts or [],
                        )
                    ]
                )
            )

            entities_text_to_ids = {
                entity_text: self.generate_id(entity_text)
                for entities_text in all_propositions_entities_texts
                for entity_text in entities_text
            }
            batch_entities_infos = [
                {
                    "entity_id": e_id,
                    "entity_label": e_text,
                }
                for e_text, e_id in entities_text_to_ids.items()
            ]

            # Create a list of dictionaries for the entities with their metadata
            batch_utterances_infos = [
                {
                    "utterance_id": self.generate_id(
                        f"{entity[0]}{entity[1]}"
                    ),  # Id of the utterance of this text label for this named entity. This is unique. But we cannot expect a 1 to 1 mapping for text label to ontology id.
                    "entity_id": entities_text_to_ids[
                        entity[1]
                    ],  # Unique id of the entity, based on the text label - for vector indexing
                    "ontology_id": entity[
                        0
                    ],  # The ontology id of the entity, which is the id of the entity in the ontology
                }
                for entity in all_distinct_pairs
            ]

            # To make the id unique, we add the chroma_collection_entities.name to the id
            batch_entities_ont_ids = list(batch_dict_of_entities.keys())
            n_entities = len(batch_entities_ont_ids)

            # we store all the info of ALL the entities, even those that not cannot added in the collection because already present
            batch_emap_infos = [
                {
                    "ontology_id": batch_entities_ont_ids[i],
                    "propositions_ids": [
                        batch_proposition_ids[k]
                        for k in batch_dict_of_entities[batch_entities_ont_ids[i]]
                    ],
                }
                for i in range(n_entities)
            ]

            # Convert entities text and entities ids to jsonlist
            for proposition in batch_propositions:
                proposition.metadata["entities_text"] = json.dumps(
                    proposition.metadata.get("entities_text")
                )
                proposition.metadata["entities_ids"] = json.dumps(
                    proposition.metadata.get("entities_ids")
                )

            ## INSERTIONS
            # Now we need to insert the propositions in the collection
            batch_propositions_texts_embeddings = (
                self.storage.document_store._embedding_fn(batch_propositions_texts)
            )
            self.logger.debug("Embeddings done.")
            self.logger.debug("Inserting propositions in the collection.")
            self.storage.document_store.add(
                documents=batch_propositions_texts,
                embeddings=batch_propositions_texts_embeddings,
                metadatas=[proposition.metadata for proposition in batch_propositions],
                ids=batch_proposition_ids,
            )
            collection_new_count = self.storage.document_store.count()
            self.logger.info(f"Proposition collection new size: {collection_new_count}")

            # Now insert the entities in the collection
            to_insert_entities_labels = [
                e_info["entity_label"] for e_info in batch_entities_infos
            ]
            to_insert_entities_ids = [
                e_info["entity_id"] for e_info in batch_entities_infos
            ]

            # Now first we insert the new entities
            self.logger.debug("Calling embeddings function on the entities.")
            to_insert_entities_embeddings = self.storage.entity_store._embedding_fn(
                to_insert_entities_labels
            )
            self.logger.debug("Embeddings done.")

            self.logger.debug("Inserting entities in the collection")
            previous_count = self.storage.entity_store.count()
            self.storage.entity_store.add(
                documents=to_insert_entities_labels,
                embeddings=to_insert_entities_embeddings,
                ids=to_insert_entities_ids,
            )

            self.storage.sqlite_db.insert_propositions(
                batch_propositions, self.storage.document_store.name()
            )

            self.storage.sqlite_db.insert_entities(
                batch_entities_infos,
                batch_emap_infos,
                batch_utterances_infos,
            )

            self.logger.debug("Loading done.")
            self.logger.info("Now we need to update the graph.")

            # Now update the graph
            self.update_graph(batch_proposition_ids)
            self.logger.info("Graph updated.")

    def update_graph(self, proposition_ids: List[str]) -> None:
        proposition2entities = (
            self.storage.sqlite_db.get_ontology_ids_by_proposition_ids(proposition_ids)
        )

        # Step 1: Get unique proposition and entity IDs
        propositions_ids = list(proposition2entities.keys())
        entities_ids = sorted(
            set(val for values in proposition2entities.values() for val in values)
        )

        # Step 2: Create mappings for keys and values
        proposition_to_index = {key: idx for idx, key in enumerate(propositions_ids)}
        entity_to_index = {value: idx for idx, value in enumerate(entities_ids)}

        # Convert the proposition2entities dictionary to an adjacency matrix p -> c
        p2e = from_dict_to_adjacency_matrix(
            proposition2entities, proposition_to_index, entity_to_index
        )

        # Step 3: Get the linked propositions for each entity c -> p
        linked_propositions = self.storage.sqlite_db.get_propositions_by_ontology_ids(
            ontology_ids=entities_ids
        )
        neighbour_proposition_ids = sorted(
            set(val for values in linked_propositions.values() for val in values)
        )
        neighbour_proposition_to_index = {
            value: idx for idx, value in enumerate(neighbour_proposition_ids)
        }

        e2p = from_dict_to_adjacency_matrix(
            linked_propositions, entity_to_index, neighbour_proposition_to_index
        )

        # Apply dot product to get all the p -> p connections with the weights
        w = np.dot(p2e, e2p)

        # remove the self-loops
        for i in range(w.shape[0]):
            for j in np.where(w[i])[0]:
                if propositions_ids[i] == neighbour_proposition_ids[j]:
                    w[i, j] = 0

        # now get the list of edges
        edges = []
        for i in range(w.shape[0]):
            for j in np.where(w[i])[0]:
                edges.append(
                    (propositions_ids[i], neighbour_proposition_ids[j], int(w[i, j]))
                )

        # Now we need to insert the edges in the sqlite db
        self.storage.sqlite_db.insert_edges(edges)
