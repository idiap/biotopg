# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import hashlib
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from typing import Dict, List

import backoff
import requests
from diskcache import Cache
from langchain_core.documents.base import Document
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from biotopg.builders.llm_modules.biohypergraph_extractor import (
    BioHyperPropositionizerWithNER,
)
from biotopg.utils.biomodels import BioEntity, BioHyperPropositionList


class IdGenerator:
    """
    A class that generates unique identifiers for documents and assigns them
    to a specified metadata key.
    Attributes:
        key_name (str): The name of the metadata key where the generated
                        unique identifier will be stored.
    Methods:
        modify(documents: List[Document]) -> None:
            Iterates over a list of documents and assigns a unique identifier
            to each document's metadata under the specified key.
    """

    def __init__(self, key_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_name = key_name

    def modify(self, documents: List[Document]) -> None:
        for i, doc in tqdm(
            enumerate(documents), desc="Generating ids", unit="documents"
        ):
            normalized_text = doc.page_content.strip()
            unique_key_id = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            doc.metadata[self.key_name] = unique_key_id


class BioPassagesToHyperpropositions:
    def __init__(
        self,
        hyperpropositionizer,
        logger=None,
        passage_id_key="passage_id",
        cache_dir="cache/propositions",
        max_workers=10,
        hyperpropositionizer_args={},
        include_supplementary_entities=True,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.hyperpropositionizer = hyperpropositionizer
        self.passage_id_key = passage_id_key
        self.hyperpropositionizer_args = hyperpropositionizer_args
        self.cache = Cache(cache_dir)
        self.max_workers = max_workers
        self.include_supplementary_entities = include_supplementary_entities
        self.logger.info(
            f"Initialized BioPassagesToHyperpropositions with cache at {cache_dir}"
        )
        self.logger.info(
            f"Include supplementary entities: {self.include_supplementary_entities}"
        )

    def _call_llm(self, doc):
        return self.hyperpropositionizer.extract_propositions(
            doc, **self.hyperpropositionizer_args
        )

    def _process_doc(self, doc):
        passage_id = doc.metadata[self.passage_id_key]
        if passage_id in self.cache:
            # Cache hit
            cached_data = self.cache[passage_id]
            # We also check that it is not empty - if it is we want to retry
            if len(cached_data["biohyperpropositions"]):
                return passage_id, cached_data

        # Cache miss: call LLM
        result = self._call_llm(doc)
        dumped = result.model_dump()
        self.cache[passage_id] = dumped
        return passage_id, dumped

    def split_documents(self, documents):
        new_documents = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_doc, doc): doc for doc in documents
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                doc = futures[future]
                try:
                    passage_id, result_data = future.result(timeout=300)
                    biohyperpropositions_list = BioHyperPropositionList(**result_data)

                    for h in biohyperpropositions_list.biohyperpropositions:
                        if not self.include_supplementary_entities:
                            h.entities = [
                                bio_entity
                                for bio_entity in h.entities
                                if not bio_entity.id.startswith("Unknown")
                            ]
                        if not h.entities:
                            self.logger.warning(
                                f"No entities found for proposition: {h.proposition}"
                            )
                            continue
                        entities_texts = [e.text for e in h.entities]
                        entities_ids = [e.id for e in h.entities]
                        new_documents.append(
                            Document(
                                page_content=h.proposition,
                                metadata={
                                    "entities_text": entities_texts,
                                    "entities_ids": entities_ids,
                                    self.passage_id_key: passage_id,
                                },
                            )
                        )
                except TimeoutError:
                    self.logger.error(f"Timeout processing {doc.metadata}")
                except Exception as e:
                    self.logger.error(f"Error processing {doc.metadata}: {e}")

        return new_documents

    def clean_cache(self, confirm: bool = True):
        """
        Clears the disk cache. Optionally ask for confirmation.
        """
        if confirm:
            user_input = input(
                "Are you sure you want to clear the cache? (y/n): "
            ).lower()
            if user_input != "y":
                print("Aborted cache cleanup.")
                return

        self.cache.clear()
        print("Cache cleared.")

    def export_cache_to_json(self, output_path: str):
        """
        Export current cache contents to a JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        serialized = {}
        for k in self.cache.iterkeys():
            serialized[k] = self.cache.get(k)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=4)

        self.logger.info(f"Cache exported to {output_path}")


class PubTatorClient:
    def __init__(self, base_url: str, max_retries: int = 3):
        self.session = requests.Session()
        self.base_url = base_url
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_tries=3
    )
    def fetch_publications(self, pmids: List[str]) -> dict:
        params = {"pmids": ",".join(pmids)}

        response = self.session.get(self.base_url, params=params, timeout=300)
        response.raise_for_status()
        return response.json()


class PubMedArticleProcessor:
    def __init__(self, logger=None, config={}):
        self.logger = logger or logging.getLogger(__name__)

        self.passage_id_generator = IdGenerator(key_name="passage_id")
        self.bioproposition_id_generator = IdGenerator(key_name="id")

        self.pubtator_chunk_size = config.get("pubtator_chunk_size", 20)
        self.pubtator_client = PubTatorClient(
            base_url="https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
        )

        self.biohyperpropositionizer = BioHyperPropositionizerWithNER(
            logger=self.logger,
            extractor_demonstration_file=config.get(
                "extractor_demonstration_file", None
            ),
        )

        self.BioPassagesToHyperpropositions = BioPassagesToHyperpropositions(
            hyperpropositionizer=self.biohyperpropositionizer,
            cache_dir=config.get("cache_dir", "cache/propositions"),
            max_workers=config.get("max_workers", 5),
            include_supplementary_entities=config.get(
                "include_supplementary_entities", True
            ),
        )

        # Here we load the excluded ontology IDs if provided in the config
        self.excluded_ontology_ids_path = config.get("excluded_ontology_ids", None)
        self.excluded_ontology_ids = set()
        if self.excluded_ontology_ids_path:
            with open(self.excluded_ontology_ids_path, "r") as f:
                self.excluded_ontology_ids = set(json.load(f))
        self.logger.info(
            f"Excluded ontology IDs loaded: {len(self.excluded_ontology_ids)}"
        )

    def request_pubtator(self, pmids: list[str]) -> list[dict]:
        all_responses = []
        for i in tqdm(
            range(0, len(pmids), self.pubtator_chunk_size),
            desc="Fetching PubTator data",
        ):
            chunk = pmids[i : i + self.pubtator_chunk_size]
            try:
                response = self.pubtator_client.fetch_publications(chunk)
                all_responses.append(response)
            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Failed to fetch data from PubTator API for chunk {chunk}: {e}"
                )
            sleep(0.5)
        return all_responses

    def _process_pubtator_response(self, response: dict):
        doc_text = "\n".join([passage["text"] for passage in response["passages"]])
        pmid = response["id"]
        biomedical_entities = defaultdict(set)
        for passage in response["passages"]:
            annotations = passage["annotations"]
            for annotation in annotations:
                text = annotation["text"]
                annot_type = annotation["infons"].get("type", "Unknown")
                identifier = annotation["infons"].get(
                    "identifier", hashlib.sha256(text.encode("utf-8")).hexdigest()
                )
                # Sometimes, there can be an entry for identifier, but it is actaully not determined and refer the '-'.
                if identifier == "-":
                    identifier = hashlib.sha256(text.encode("utf-8")).hexdigest()

                # Add the type to the identifier
                identifier = f"{annot_type}|{identifier}"

                # Check if the identifier is in the excluded ontology IDs
                if identifier in self.excluded_ontology_ids:
                    continue

                if text.lower() not in (
                    t.lower() for t in biomedical_entities[identifier]
                ):
                    biomedical_entities[identifier].add(text)

        doc = Document(
            page_content=doc_text,
            metadata={
                "doc_id": pmid,
                "collection": "PubMed",
                "entities": biomedical_entities,
            },
        )
        return doc

    def get_tiabs_passages(self, pmids: List[str]):
        pubtator_responses = self.request_pubtator(pmids)
        passages = [
            self._process_pubtator_response(doc)
            for pubtator_response in pubtator_responses
            for doc in pubtator_response["PubTator3"]
        ]

        # Here the passages are created
        self.passage_id_generator.modify(passages)

        return passages

    def get_biohyperpropositions(self, bio_passages: List[Document]) -> List[Document]:
        """
        Extract hyperpropositions from the passages.
        """
        # Apply the passage to hyperproposition conversion
        biohyperpropositions = self.BioPassagesToHyperpropositions.split_documents(
            bio_passages
        )

        # Modify the hyperproposition IDs
        self.bioproposition_id_generator.modify(biohyperpropositions)

        return biohyperpropositions
