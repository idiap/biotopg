# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union, cast

import chromadb
from chromadb import Collection, Documents, Embeddings
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from biotopg.builders.sql_db import MapDB


class ChromaEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        encoding_params: dict = {},
        device: str = "cpu",
        **kwargs: Any,
    ):
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self.encoding_params = encoding_params

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, **self.encoding_params).tolist()

    def __call__(self, input: Documents) -> Embeddings:
        return cast(Embeddings, self.encode(input))


class ChromaEmbeddingStorage:
    def __init__(
        self,
        path: str,
        collection_name: str,
        embedding_fn,
        logger: Optional[logging.Logger] = None,
    ):
        self.path = path
        self.collection_name = collection_name
        self._embedding_fn = embedding_fn
        self.logger = logger or logging.getLogger(__name__)
        self._collection = self._init_collection()

    def _init_collection(self) -> Collection:
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.logger.debug(f"Created Chroma DB directory: {self.path}")

        client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(allow_reset=False),
        )

        self.logger.debug(f"Getting or creating collection: {self.collection_name}")
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 128,
                "hnsw:M": 16,
            },
        )

    def name(self) -> str:
        return self._collection.name

    def count(self) -> int:
        return self._collection.count()

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        self.logger.debug(
            f"Adding {len(documents)} documents to collection '{self.name()}'"
        )
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def get(self, ids: List[str], include: Optional[List[str]] = None) -> dict:
        self.logger.debug(
            f"Retrieving {len(ids)} documents from collection '{self.name()}'"
        )
        return self._collection.get(ids=ids, include=include)

    def query(
        self,
        query_embeddings: Union[List[float], List[List[float]]],
        n_results: int = 5,
        include: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> dict:
        if isinstance(query_embeddings[0], float):
            query_embeddings = [query_embeddings]
        self.logger.debug(
            f"Querying collection '{self.name()}' for top {n_results} results"
        )
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
            ids=ids,
        )


class HybridStore:
    def __init__(self, config: Mapping[str, Any], logger=None):
        self.config = config
        self.document_store = None
        self.entity_store = None
        self.sqlite_db = None
        self.logger = logger or logging.getLogger(__name__)
        self.init_storage()

    def init_storage(self):
        self.logger.info("Initializing HybridStore")

        # ----- Embedding Parameters -----
        default_encoding_params = {
            "batch_size": 128,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }

        # ----- Embedding Functions -----
        self.logger.info("Initializing embedding functions for documents and entities")
        document_embedding_fn = ChromaEmbeddingFunction(
            model_name=self.config.get(
                "model_name_documents", "BAAI/bge-large-en-v1.5"
            ),
            encoding_params=self.config.get("encoding_params", default_encoding_params),
            device=self.config.get("device", "cpu"),
            trust_remote_code=True,
        )

        entity_embedding_fn = ChromaEmbeddingFunction(
            model_name=self.config.get(
                "model_name_entities", "sentence-transformers/all-mpnet-base-v2"
            ),
            encoding_params=self.config.get("encoding_params", default_encoding_params),
            device=self.config.get("device", "cpu"),
        )

        # ----- Chroma Document Store -----
        self.logger.info("Initializing Chroma document store")
        self.document_store = ChromaEmbeddingStorage(
            path=self.config.get("document_chromadb_path", "./hybrid-db/chroma-docs"),
            collection_name=self.config.get("collection_name", "test"),
            embedding_fn=document_embedding_fn,
            logger=self.logger,
        )

        # ----- Chroma Entity Store -----
        self.logger.info("Initializing Chroma entity store")
        self.entity_store = ChromaEmbeddingStorage(
            path=self.config.get(
                "entities_chromadb_path", "./hybrid-db/chroma-entities"
            ),
            collection_name=self.config.get("collection_name", "test"),
            embedding_fn=entity_embedding_fn,
            logger=self.logger,
        )

        # ----- SQLite DB -----
        self.logger.info("Initializing SQLite database")
        # TODO: MapDB should also be abstracted to allow different implementations
        self.sqlite_db = MapDB(
            db_path=self.config.get("sqlite_db_path", "./hybrid-db/sqlite_db.db"),
            logger=self.logger,
        )

        self.logger.info("HybridStore initialized successfully")

    def show_statistics(self) -> None:
        # self.sqlite_db.show_db_statistics()
        self.logger.info(f"Document collection size: {self.document_store.count()}")
        self.logger.info(f"Entities collection size: {self.entity_store.count()}")

    def export_all_passages(self, output_path: str) -> None:
        """
        Exports all passages from the document store.
        """
        self.logger.info("Exporting all passages from the store")
        passages = self.sqlite_db.get_all_passages()

        # Save as json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Exported {len(passages)} passages to {output_path}")

    def export_all_hyperpropositions(self, output_path: str) -> None:
        self.logger.info("Exporting all hyperpropositions from the store")
        hyperpropositions = self.document_store._collection.get()

        all_hyperpropositions = [
            {"id": _id, "page_content": doc, "metadata": meta}
            for _id, doc, meta in zip(
                hyperpropositions["ids"],
                hyperpropositions["documents"],
                hyperpropositions["metadatas"],
            )
        ]
        for hyperproposition in all_hyperpropositions:
            hyperproposition["metadata"]["entities_ids"] = json.loads(
                hyperproposition["metadata"].get("entities_ids", "[]")
            )
            hyperproposition["metadata"]["entities_text"] = json.loads(
                hyperproposition["metadata"].get("entities_text", "[]")
            )
        # Save as json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_hyperpropositions, f, ensure_ascii=False, indent=4)

        self.logger.info(
            f"Exported {len(all_hyperpropositions)} passages to {output_path}"
        )
