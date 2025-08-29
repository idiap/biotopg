# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT


import json
import os
from logging import DEBUG, ERROR, INFO, WARNING
from typing import Any, Dict, List, Mapping

import dspy
import yaml
from colorama import init
from langchain_core.documents.base import Document

from biotopg.builders.bio_hybrid_vector_concept_db import BioHybridBaseLoader
from biotopg.builders.biochuncking import PubMedArticleProcessor
from biotopg.builders.storage import HybridStore
from biotopg.rag.hybrid_rag import HybridRetrieval
from biotopg.rag.query_manager import QueryManager
from biotopg.utils.llm import get_cost, load_and_configure_llm
from biotopg.utils.logger import get_std_logger


class Biotopg:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self.logger = get_std_logger(**self.config.get("logger_params", {}))
        self.logger.info("Initializing BioTopg with configuration")

        self.document_processor = PubMedArticleProcessor(
            self.logger, config=self.config.get("pubmed_processor_params", {})
        )
        self.store = HybridStore(
            config=self.config.get("storage_params", {}), logger=self.logger
        )

        self.logger.info("Initializing the base loader")
        self.loader = BioHybridBaseLoader(
            storage=self.store,
            config=self.config.get("loaders_params", {}),
            logger=self.logger,
        )
        self.logger.info("Initializing the retriever")
        self.retriever = HybridRetrieval(store=self.store, logger=self.logger)

        self.logger.info("Initializing the query manager")
        self.query_manager = QueryManager(
            retriever=self.retriever,
            logger=self.logger,
            max_query_workers=self.config.get("llm_config", {}).get(
                "max_query_workers", 1
            ),
        )

        # Load the LLM
        self.logger.info("Loading the LLM config")
        self.lm = load_and_configure_llm(llm_config=self.config.get("llm_config", {}))
        # Initialize colorama
        init(autoreset=True)
        self.logger.info("All initializations ok!")

    def insert_pmids(self, pmids: List[str]) -> None:
        """
        Inserts a text into the document processor and returns the processed passages.
        """

        # Get the TiAb passages from PubTator3
        passages = self.document_processor.get_tiabs_passages(pmids)

        # Index the passages
        self.loader.index_passages(passages)

        # Extract hyperpropositions from passages
        biohyperpropositions = self.document_processor.get_biohyperpropositions(
            passages
        )

        # While duplicated ids will not be added - we should prevent duplicated ids in the same batch insert ! - We keep the first one
        all_ids = []
        filtered_hyperpropositions = []
        for biohyperproposition in biohyperpropositions:
            if biohyperproposition.metadata["id"] not in all_ids:
                all_ids.append(biohyperproposition.metadata["id"])
                filtered_hyperpropositions.append(biohyperproposition)

        # Index the hyperpropositions
        self.loader.index_propositions(filtered_hyperpropositions)

        cost = get_cost(self.lm)
        self.logger.info(f"Total cost of LLM calls: {cost:.4f} USD")

    def query(
        self,
        question: str,
        mode: str = "local",
        max_iter: int = 5,
        retriever_args: dict = {},
        *args,
        **kwargs,
    ) -> Any:
        answer = self.query_manager.query(
            question=question,
            mode=mode,
            max_iter=max_iter,
            retriever_args=retriever_args,
            *args,
            **kwargs,
        )
        return answer

    def show_reasoning(self):
        """
        Displays the reasoning of the last query.
        """
        self.query_manager.show_reasoning()

    def get_qa_memory(self):
        return self.query_manager.get_memory()

    def load_config(self, config_path: str) -> Mapping[str, Any]:
        """
        Loads the configuration from a YAML file.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def initialize(base_path: str, collection_name: str = "PubMed") -> str:
        """
        Initializes a new project structure with default configuration.

        Args:
            base_path (str): The directory where the project should be created.
            collection_name (str): Name of the collection (default: "BioASQ").

        Returns:
            str: Path to the generated config.yaml file.
        """
        project_dir = os.path.join(base_path, collection_name)
        dbs_dir = os.path.join(project_dir, "dbs")
        logs_dir = os.path.join(project_dir, "logs_and_cache")

        if os.path.exists(project_dir):
            raise FileExistsError(
                f"Project directory '{project_dir}' already exists. Please choose a different name or if you want to create an other db."
            )

        # Create directories
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(dbs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Default configuration
        config: Dict[str, Any] = {
            "logger_params": {
                "name": collection_name,
                "path": logs_dir,
                "stdout": True,
                "level": "INFO",
            },
            "pubmed_processor_params": {
                "pubtator_chunk_size": 20,
                "extractor_demonstration_file": None,
                "cache_dir": logs_dir,
                "excluded_ontology_ids": None,
                "include_supplementary_entities": True,
                "max_workers": 6,
            },
            "storage_params": {
                "document_chromadb_path": os.path.join(dbs_dir, "chromadb-docs"),
                "entities_chromadb_path": os.path.join(dbs_dir, "chromadb-entities"),
                "sqlite_db_path": os.path.join(dbs_dir, "sqlite_db.db"),
                "collection_name": collection_name,
                "device": "cuda",
                "model_name_documents": "BAAI/bge-large-en-v1.5",
                "model_name_entities": "BAAI/bge-large-en-v1.5",
                "encoding_params": {
                    "batch_size": 128,
                    "convert_to_numpy": True,
                    "show_progress_bar": True,
                },
            },
            "loaders_params": {
                "chroma_loading_batch_size": 512,
            },
            "llm_config": {
                "api_base": None,
                "llm_name": "openai/gpt-4o-mini",
                "max_tokens": 2048,
                "max_query_workers": 5,
            },
        }

        # Save config to YAML
        config_path = os.path.join(project_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return config_path
