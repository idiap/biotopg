# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: MIT

import argparse
import json
import os
import time

import yaml

from biotopg.biotopg import Biotopg

parser = argparse.ArgumentParser(
    description="Populate collection with documents and export results."
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the input JSON file containing documents.",
)
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the configuration YAML file.",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    required=True,
    help="Directory to store cache and exported files.",
)

args = parser.parse_args()

input_file = args.input_file
config_path = args.config_path
cache_dir = args.cache_dir

all_pmids = json.loads(open(input_file, "r").read())
n_pmids = len(all_pmids)

print(f"Number of PMIDs to insert: {n_pmids}")

with open(config_path, "r") as file:
    system_config = yaml.safe_load(file)

system = Biotopg(config=system_config)

# A simple test
system.lm("This is a test to check if you are ready. Just answer 'I am ready'.")

MAX_BATCH_SIZE = 200

for i in range(0, len(all_pmids), MAX_BATCH_SIZE):
    batch_pmids = all_pmids[i : i + MAX_BATCH_SIZE]
    print(f"Processing batch {i // MAX_BATCH_SIZE + 1} with {len(batch_pmids)} PMIDs")
    start_time = time.time()
    system.insert_pmids(pmids=batch_pmids)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Time taken: {elapsed_time:.4f} seconds for batch processing of {len(batch_pmids)} PMIDs"
    )

system.store.show_statistics()


print("Exporting hyperpropositions and passages...")
system.store.export_all_hyperpropositions(
    os.path.join(cache_dir, "hyperpropositions.json")
)
system.store.export_all_passages(os.path.join(cache_dir, "passages.json"))


print("Hyperpropositions and passages exported successfully.")
