# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import logging
import os
import sqlite3
from collections import defaultdict
from itertools import combinations
from typing import Any, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


class MapDB:
    """
    Manage a SQLite database to store and retrieve passages, propositions, and entities.
    """

    def __init__(self, db_path, logger=None, chunk_size=60000, verbose=False):
        """
        Initialize the database connection, create tables if they do not exist,
        and set up logging.

        :param db_path: Path to the SQLite database file.
        :param log_file_path: Path to the log file (default is 'db_operations.log').
        """
        self.db_path = db_path
        self.verbose = verbose
        self.chunk_size = chunk_size
        # Check that sqlite_db_path directory exists
        sqlite_db_dir = os.path.dirname(db_path)
        if not os.path.exists(sqlite_db_dir):
            os.makedirs(sqlite_db_dir)

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Set up logging with the provided log file path
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"Connecting to database at {db_path}")

        self._create_tables()

    def _create_tables(self):
        """Create the passages and propositions tables if they do not exist."""
        try:
            self.logger.info("Creating tables if they do not exist.")
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS passages (
                passage_id TEXT PRIMARY KEY,
                page_content TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                collection TEXT NOT NULL
            );
            """
            )

            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS propositions (
                id TEXT PRIMARY KEY,
                passage_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                FOREIGN KEY (passage_id) REFERENCES passages (passage_id)
            );
            """
            )
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                label TEXT NOT NULL
            );
            """
            )

            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS utterances (
                utterance_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                ontology_id TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
            );
            """
            )

            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS emap (
                emap_id TEXT PRIMARY KEY,
                ontology_id TEXT NOT NULL,
                proposition_id TEXT NOT NULL,
                FOREIGN KEY (proposition_id) REFERENCES propositions (id)
            );
            """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_neighbors (
                    node_id INTEGER,
                    neighbor_id INTEGER,
                    weight REAL,
                    PRIMARY KEY (node_id, neighbor_id),
                    FOREIGN KEY (node_id) REFERENCES propositions(id),
                    FOREIGN KEY (neighbor_id) REFERENCES propositions(id)
                );
            """
            )
            self.conn.commit()
            self.logger.info("Tables are created or already exist.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def insert_passages(self, passages, collection):
        """
        Insert multiple passage entries into the passages table using batch insertions.

        :param passages: List of passage objects with `.page_content` and `.metadata`.
        :param collection: The name of the collection to which the passages belong.
        """

        insert_data = [
            (
                passage.metadata["passage_id"],
                passage.page_content,
                passage.metadata["doc_id"],
                collection,
            )
            for passage in passages
        ]

        self.logger.info(f"Inserting {len(insert_data)} passages into the database.")
        try:
            with self.conn:  # Ensures commit/rollback automatically
                for batch in tqdm(
                    self.chunk_list(insert_data, self.chunk_size),
                    desc="Inserting passages",
                    unit="batch",
                ):
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO passages (passage_id, page_content, doc_id, collection)
                        VALUES (?, ?, ?, ?)
                        """,
                        batch,
                    )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting passages: {e}")
            raise
        self.logger.info("All passages inserted successfully.")

    def insert_propositions(self, propositions, collection):
        """
        Insert propositions mapping to link proposition IDs to passage IDs.

        :param propositions: List of proposition objects containing `.metadata["id"]` and `.metadata["passage_id"]`.
        :param collection: The name of the collection to which the propositions belong.
        """

        insert_data = [
            (prop.metadata["id"], prop.metadata["passage_id"], collection)
            for prop in propositions
        ]

        self.logger.info(
            f"Inserting {len(insert_data)} propositions into the database."
        )
        try:
            with self.conn:  # Ensures commit or rollback as needed
                for batch in tqdm(
                    self.chunk_list(insert_data, self.chunk_size),
                    desc="Inserting propositions",
                    unit="batch",
                ):
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO propositions (id, passage_id, collection)
                        VALUES (?, ?, ?)
                        """,
                        batch,
                    )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting propositions: {e}")
            raise

        self.logger.info("All propositions inserted successfully.")

    def insert_text_entities(self, entities):
        """
        Batch insert entities into the 'entities' table.

        :param entities: List of dictionaries with keys "entity_id" and "entity_label".
        :param collection: Name of the collection the entities belong to.
        """

        insert_data = [
            (entity["entity_id"], entity["entity_label"]) for entity in entities
        ]

        self.logger.info(f"Inserting {len(insert_data)} entities.")
        try:
            with self.conn:  # Ensures all operations are in a single transaction
                for batch in tqdm(
                    self.chunk_list(insert_data, self.chunk_size),
                    desc="Inserting entities",
                    unit="batch",
                ):
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO entities (entity_id, label)
                        VALUES (?, ?)
                        """,
                        batch,
                    )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting entities: {e}")
            raise
        self.logger.info("All entities inserted successfully.")

    def insert_emap(self, emap):
        """
        Insert entity-to-proposition mappings into the 'emap' table.

        Each mapping must include matching lengths of 'propositions_id' and 'passages_id'.

        :param entities: List of dictionaries with keys: 'ontology_id', 'propositions_id', and 'passages_id'.
        """

        # Prepare all rows in memory before inserting
        insert_data = [
            (
                f"{entity['ontology_id']}{prop_id}",
                entity["ontology_id"],
                prop_id,
            )
            for entity in emap
            for prop_id in entity["propositions_ids"]
        ]

        self.logger.info(f"Inserting {len(insert_data)} entity mappings.")
        try:
            with self.conn:  # Transaction block
                for batch in tqdm(
                    self.chunk_list(insert_data, self.chunk_size),
                    desc="Inserting entity mapping",
                    unit="batch",
                ):
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO emap (emap_id, ontology_id, proposition_id)
                        VALUES (?, ?, ?)
                        """,
                        batch,
                    )
        except AssertionError as ae:
            self.logger.error(f"Data shape error: {ae}")
            raise
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting entity mappings: {e}")
            raise
        self.logger.info("All entity mappings inserted successfully.")

    def insert_utterances(self, utterances):
        """
        Insert utterances into the 'utterances' table.

        :param utterances: List of dictionaries with keys 'utterance_id', 'entity_id', and 'ontology_id'.
        """

        insert_data = [
            (
                utterance["utterance_id"],
                utterance["entity_id"],
                utterance["ontology_id"],
            )
            for utterance in utterances
        ]

        self.logger.info(f"Inserting {len(insert_data)} utterances.")
        try:
            with self.conn:  # Ensures commit or rollback as needed
                for batch in tqdm(
                    self.chunk_list(insert_data, self.chunk_size),
                    desc="Inserting utterances",
                    unit="batch",
                ):
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO utterances (utterance_id, entity_id, ontology_id)
                        VALUES (?, ?, ?)
                        """,
                        batch,
                    )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting utterances: {e}")
            raise
        self.logger.info("All utterances inserted successfully.")

    def insert_entities(self, entities, emap, utterances):
        # insert data in the entities table
        self.insert_text_entities(entities)
        # insert data in the emap table
        self.insert_emap(emap)
        # insert data in the utterances table
        self.insert_utterances(utterances)

    def chunk_list(self, lst, size):
        """Yield successive n-sized chunks from the input list."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    def insert_edges(self, edges: List[Tuple[str, str, float]]) -> None:
        """ """

        # Duplicate each edge for undirected representation
        insert_data = edges

        self.logger.info(
            f"Inserting {len(insert_data)} total edge entries (undirected) with chunk size {self.chunk_size}. - Some may be duplicates."
        )
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=OFF;")
            self.conn.execute("PRAGMA foreign_keys=OFF;")
            with self.conn:  # Transaction block (auto commit or rollback)
                for batch in tqdm(
                    self.chunk_list(insert_data, self.chunk_size),
                    desc="Inserting edges",
                    unit="batch",
                ):
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO edge_neighbors (node_id, neighbor_id, weight)
                        VALUES (?, ?, ?)
                        """,
                        batch,
                    )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting edges: {e}")
            raise
        finally:
            # Restore safe defaults
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute("PRAGMA synchronous=FULL;")
        self.logger.info("All edges inserted successfully.")

    def get_all_edges(self) -> List[Tuple[str, str, float]]:
        """
        Retrieve all edges from the 'edge_neighbors' table.

        :return: A list of tuples where each tuple represents an edge (node_id, neighbor_id, weight).
        """
        try:
            self.logger.info("Fetching all edges from the database.")
            query = """
                SELECT node_id, neighbor_id, weight
                FROM edge_neighbors
            """
            self.cursor.execute(query)
            edges = self.cursor.fetchall()
            self.logger.info(f"Fetched {len(edges)} edges from the database.")
            return edges
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching edges: {e}")
            raise

    def get_ontology_degrees(self) -> List[Tuple[str, int]]:
        """
        Retrieve the degree (number of propositions) associated with each ontology_id from the 'emap' table.

        :return: A list of tuples where each tuple represents an ontology_id and its degree.
        """
        ontology_degrees = []
        try:
            self.logger.info("Fetching ontology degrees from the database.")
            query = """
                SELECT ontology_id, COUNT(proposition_id) as degree
                FROM emap
                GROUP BY ontology_id
            """
            self.cursor.execute(query)
            ontology_degrees = self.cursor.fetchall()
            self.logger.info(
                f"Fetched degrees for {len(ontology_degrees)} ontology IDs."
            )
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching ontology degrees: {e}")
            raise

        ontology_degrees.sort(key=lambda x: x[1], reverse=True)
        return ontology_degrees

    def get_passages_by_proposition_ids(self, proposition_ids):
        """
        Fetch all passages related to a list of proposition ids, using batching for large requests.

        :param proposition_ids: List of proposition ids to fetch passages for.
        :return: A dictionary where keys are proposition ids and values are associated passages with metadata.
        """
        if not proposition_ids:
            self.logger.warning("No proposition ids provided.")
            return {}

        result = {}

        batch_iter = self.chunk_list(proposition_ids, self.chunk_size)

        # Iterate over chunks to handle large lists of IDs
        for batch in tqdm(batch_iter, desc="Fetching passages", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT pr.id, p.passage_id, p.page_content, p.doc_id, p.collection
                FROM propositions pr
                JOIN passages p ON p.passage_id = pr.passage_id
                WHERE pr.id IN ({placeholders})
            """

            try:
                # Execute the query with the current batch of proposition_ids
                self.cursor.execute(query, batch)
                rows = self.cursor.fetchall()
            except sqlite3.Error as e:
                self.logger.error(f"Error fetching passages: {e}")
                return {}

            # Map the results to a dictionary based on proposition id
            for row in rows:
                proposition_id, passage_id, page_content, doc_id, collection = row
                passage = {
                    "passage_id": passage_id,
                    "page_content": page_content,
                    "doc_id": doc_id,
                    "collection": collection,
                }
                result[proposition_id] = passage

        self.logger.info(
            f"Fetched {len(result)} passages for {len(proposition_ids)} proposition ids."
        )
        return result

    def get_propositions_by_ontology_ids(self, ontology_ids):
        """
        Fetch all proposition IDs related to a list of entity IDs, using batching for large requests.

        :param ontology_ids: List of entity IDs to fetch proposition IDs for.
        :return: A dictionary where keys are entity IDs and values are lists of proposition IDs.
        """
        if not ontology_ids:
            self.logger.warning("No entity IDs provided.")
            return {}

        result = defaultdict(set)
        batch_iter = self.chunk_list(ontology_ids, self.chunk_size)

        for batch in tqdm(batch_iter, desc="Fetching propositions", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT ontology_id, proposition_id
                FROM emap
                WHERE ontology_id IN ({placeholders})
            """

            try:
                self.cursor.execute(query, batch)
                for ontology_id, proposition_id in self.cursor.fetchall():
                    result[ontology_id].add(proposition_id)

            except sqlite3.Error as e:
                self.logger.error(f"Error fetching propositions: {e}")
                return {}

        # Convert sets to lists and defaultdict to a normal dictionary
        result = {
            ontology_id: list(propositions)
            for ontology_id, propositions in result.items()
        }

        self.logger.info(f"Fetched propositions for {len(result)} unique entity IDs.")

        return result

    def get_ontology_ids_by_proposition_ids(self, proposition_ids):
        """
        Fetch all entity IDs and labels related to a list of proposition IDs, using batching for large requests.

        :param proposition_ids: List of proposition IDs to fetch entity IDs for.
        :return: A dictionary where keys are proposition IDs and values are lists of ontology ids.
        """
        if not proposition_ids:
            self.logger.warning("No proposition IDs provided.")
            return {}

        result = {}
        batch_iter = self.chunk_list(proposition_ids, self.chunk_size)

        for batch in tqdm(batch_iter, desc="Fetching entities", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT e.proposition_id, e.ontology_id
                FROM emap e
                WHERE e.proposition_id IN ({placeholders})
            """

            try:
                self.cursor.execute(query, batch)
                for proposition_id, ontology_id in self.cursor.fetchall():
                    result.setdefault(proposition_id, []).append(ontology_id)

            except sqlite3.Error as e:
                self.logger.error(f"Error fetching entities: {e}")
                return {}

        self.logger.info(f"Fetched entities for {len(result)} unique proposition IDs.")
        return result

    def get_propositions_by_entity_ids(self, entity_ids):
        """
        Fetch all proposition IDs related to a list of entity IDs, using batching for large requests.
        :param entity_ids: List of entity IDs to fetch proposition IDs for.
        :return: A dictionary where keys are entity IDs and values are lists of proposition IDs.
        """

        if not entity_ids:
            self.logger.warning("No entity IDs provided.")
            return {}

        result = defaultdict(set)
        batch_iter = self.chunk_list(entity_ids, self.chunk_size)

        for batch in tqdm(batch_iter, desc="Fetching propositions", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
            SELECT DISTINCT entity_id, proposition_id
                FROM utterances u
                JOIN emap em ON u.ontology_id = em.ontology_id
                WHERE u.entity_id IN ({placeholders});
            """
            try:
                self.cursor.execute(query, batch)
                for entity_id, proposition_id in self.cursor.fetchall():
                    result[entity_id].add(proposition_id)

            except sqlite3.Error as e:
                self.logger.error(f"Error fetching propositions: {e}")
                return {}

        # Convert sets to lists and defaultdict to a normal dictionary
        result = {
            entity_id: list(propositions) for entity_id, propositions in result.items()
        }

        self.logger.info(f"Fetched propositions for {len(result)} unique entity IDs.")

        return result

    def close(self):
        """Close the database connection."""
        try:
            self.logger.info("Closing database connection.")
            self.conn.close()
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
            raise

    def get_table_row_count(self, table_name):
        """
        Get the row count for a given table.

        :param table_name: The name of the table to query.
        :return: The number of rows in the table.
        """
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = self.cursor.fetchone()[0]
            return row_count
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving row count for table {table_name}: {e}")
            raise

    def get_db_size(self):
        """
        Get the size of the database file.

        :return: The size of the database in bytes.
        """
        try:
            db_size = os.path.getsize(self.db_path)
            return db_size
        except OSError as e:
            self.logger.error(f"Error getting size of database file: {e}")
            raise

    def show_db_statistics(self):
        """
        Display basic statistics about the database: table row counts and file size.
        """
        try:
            self.logger.info("Fetching database statistics...")

            # List of all tables to fetch statistics for
            tables = [
                "passages",
                "propositions",
                "entities",
                "utterances",
                "emap",
                "edge_neighbors",
            ]

            # Fetch row counts for each table
            table_stats = {}
            for table in tables:
                try:
                    table_stats[table] = self.get_table_row_count(table)
                except Exception as e:
                    self.logger.warning(
                        f"Could not fetch row count for table '{table}': {e}"
                    )
                    table_stats[table] = "Error"

            # Get database size
            db_size = self.get_db_size()

            # Display the statistics
            self.logger.info(f"Database Size: {db_size / 1024 / 1024:.2f} MB")
            for table, count in table_stats.items():
                self.logger.info(f"Total Rows in '{table}': {count}")

        except Exception as e:
            self.logger.error(f"Error fetching database statistics: {e}")
            raise

    def get_p2p_via_entities(self, masks_ontology_ids: List[str] = []):
        # Create a set of unique node names
        edges = self.get_all_edges()

        # Convert edges to index-based representation
        # Do we apply the mask ?
        edge_list = []
        if len(masks_ontology_ids):
            # First, get the assocaited propositions:
            excluded_edges_partners = self.get_propositions_by_ontology_ids(
                masks_ontology_ids
            )
            for ontology_id, partners in excluded_edges_partners.items():
                excluded_edges_partners[ontology_id] = set(
                    excluded_edges_partners[ontology_id]
                )

            # We have the set of all neighbours from each ontology_id link.
            all_partners_sets = list(excluded_edges_partners.values())
            for src, tgt, w in tqdm(edges, desc="Applying mask on edges", unit="edge"):
                n_link = sum(
                    [
                        src in partners_set and tgt in partners_set
                        for partners_set in all_partners_sets
                    ]
                )
                # Check if src and tgt are in the same set of partners
                new_weight = max(0, w - n_link)
                if not new_weight:
                    continue
                edge_list.append((src, tgt, (w - n_link)))
        else:
            # No mask, we keep all edges
            edge_list = edges

        # Create the graph
        g = nx.Graph()
        # Add edges with weights
        for src, tgt, weight in edges:
            g.add_edge(src, tgt, weight=weight)

        # Remove self-loops if any
        g.remove_edges_from(nx.selfloop_edges(g))

        return g

    def get_all_passages(self) -> List[dict]:
        """
        Retrieve all passages from the 'passages' table.

        :return: A list of dictionaries where each dictionary represents a passage with its metadata.
        """
        try:
            self.logger.info("Fetching all passages from the database.")
            query = """
                SELECT passage_id, page_content, doc_id, collection
                FROM passages
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            passages = [
                {
                    "passage_id": row[0],
                    "page_content": row[1],
                    "doc_id": row[2],
                    "collection": row[3],
                }
                for row in rows
            ]
            self.logger.info(f"Fetched {len(passages)} passages from the database.")
            return passages
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching passages: {e}")
            raise

    def get_all_passage_proposition_links(self) -> List[Tuple[str, str]]:
        """
        Retrieve all links between passage IDs and proposition IDs from the 'propositions' table.
        :return: A list of tuples where each tuple represents a link (passage_id, proposition_id).
        """
        try:
            self.logger.info(
                "Fetching all passage-proposition links from the database."
            )
            query = """
                SELECT passage_id, id
                FROM propositions
            """
            self.cursor.execute(query)
            links = self.cursor.fetchall()
            self.logger.info(
                f"Fetched {len(links)} passage-proposition links from the database."
            )
            return links
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching passage-proposition links: {e}")
            raise

    def get_graphs(self, masks_ontology_ids: List[str] = []):
        self.logger.info("Building graphs from the database.")
        # We have the first graph: proposition -> proposition via entities
        g_p2p_via_entities = self.get_p2p_via_entities(
            masks_ontology_ids=masks_ontology_ids
        )
        self.logger.info(
            f"Graph size via entities: {g_p2p_via_entities.number_of_nodes()} vertices, {g_p2p_via_entities.number_of_edges()} edges"
        )

        # Get all passage-proposition links
        tuples_list = self.get_all_passage_proposition_links()
        # Group phrases by passage
        passage_to_propositions = defaultdict(set)
        for passage_id, proposition_id in tuples_list:
            passage_to_propositions[passage_id].add(proposition_id)

        # Create second graph: proposition -> proposition via passage
        g_p2p_via_passage = nx.Graph()

        # Add edges between propositions that share passages
        for passage_id, proposition_ids in passage_to_propositions.items():
            # Only consider propositions that exist in the first graph
            valid_propositions = [
                prop_id
                for prop_id in proposition_ids
                if prop_id in g_p2p_via_entities.nodes()
            ]

            # Create all combinations of proposition pairs
            for prop1, prop2 in combinations(valid_propositions, 2):
                g_p2p_via_passage.add_edge(prop1, prop2, weight=1.0)

        self.logger.info(
            f"Graph size via passage: {g_p2p_via_passage.number_of_nodes()} vertices, {g_p2p_via_passage.number_of_edges()} edges"
        )

        # Create the merge graph
        g_via_entities_and_passage = g_p2p_via_entities.copy()
        self.logger.info(
            "Merging graphs: proposition -> proposition via entities and passage"
        )

        # Add edges from second graph, combining weights
        for u, v, data in g_p2p_via_passage.edges(data=True):
            if g_via_entities_and_passage.has_edge(u, v):
                # Edge exists: sum the weights
                g_via_entities_and_passage[u][v]["weight"] += data["weight"]
            else:
                # Edge doesn't exist: add new edge
                g_via_entities_and_passage.add_edge(u, v, weight=data["weight"])

        self.logger.debug(
            f"Graph size after merging: {g_via_entities_and_passage.number_of_nodes()} vertices, {g_via_entities_and_passage.number_of_edges()} edges"
        )
        self.logger.debug(
            f"Graph size after simplification: {g_via_entities_and_passage.number_of_nodes()} vertices, {g_via_entities_and_passage.number_of_edges()} edges"
        )

        return g_p2p_via_entities, g_p2p_via_passage, g_via_entities_and_passage
