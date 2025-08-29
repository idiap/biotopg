<!--
Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>

SPDX-License-Identifier: MIT
-->

# BioToPG

## Installation

## Initialization

Before anything, ensure that `dspy` is properly set-up.
```bash
DSPY_CACHE_DIR="/tmp/dspy"
mkdir -p $DSPY_CACHE_DIR
export DSPY_CACHEDIR=$DSPY_CACHE_DIR
```

## Quick Start

### Settings

Prepare a project directory

```python
from biotopg import Biotopg
Biotopg.initialize("outputs", "PubMed")
```


In the config file
```yaml
  excluded_ontology_ids: null
  include_supplementary_entities: True
```

- ```excluded_ontology_ids``` gives the path a file containing a json list of all ontology ids, to not consider as entities.
- ```include_supplementary_entities``` discard all supplementary entities extracted by the LLM.


**Warnings:**
- By default, no `extractor_demonstration_file` is specified. Specify one if you want to add demonstrations for the extraction process (recommended). There is an example in `demonstrations/demo_hyperproposition_with_ner.json.json`
- If you are using OpenAI models, specify your API_KEY in the env variable `OPENAI_API_KEY` (recommended), or, add it to the yaml file.

Initialize the system by passing the path to the config dict
```python
import yaml
config_path = "outputs/PubMed/config.yaml"
with open(config_path, "r") as file:
    system_config = yaml.safe_load(file)

system = Biotopg(config=system_config)
```

### Indexing

Insert some pubmed article using their pmids
```python
batch_pmids = ["11299965", "18218922", "16245277", "14633596", "9560222", "10797941", "23194061", "12049630", "7758945"]
system.insert_pmids(pmids=batch_pmids)
```

You can export the passages and propositions using :

```python
import os
outdir = "outputs"
system.store.export_all_hyperpropositions(
    os.path.join(outdir, "hyperpropositions.json")
)
system.store.export_all_passages(os.path.join(outdir, "passages.json"))
```

### Queries

There are 2 modes: `local` for fact-oriented queries and `global` for more abstract queries.

#### Local

```python

# if new documents have been added since, it is recommended to reload the graph. When reloading the graph.
system.retriever.load_graphs(force_reload=True)

retriever_args = {
    "initial_retriever_args": {
        "e_syn_k": 5,
        "e_syn_threshold": 0.80,
        "lambda_mmr": 1.0,
        "p_k": 20
    },
    "q": 0.5,
    "damping": 0.5,
    "cosine_threshold": 0.4,
    "horizon_threshold": 1e-4,
    "temperature": 0.1,
    "top_k": 20,
    "use_passage_links": True
}


question = "What is Tastin ?"
qa_mode = "local"
max_iter = 1
predicted_answer, documents, _memory = system.query(
    question,
    mode=qa_mode,
    max_iter=max_iter,
    retriever_args=retriever_args,
)

print("Answer:", predicted_answer)
print("Documents:")
for pmid, data in documents.items():
    print(f"PMID: {pmid}")
    print(f"Passage ID: {data['passage_id']}")
    print(f"Text: {data['text']}")
    print("Facts:")
    for fact in data["facts"]:
        print(f"- {fact}")
    print()
```

#### Global


```python
question = "What is Tastin ?"
qa_mode = "global"
max_iter = 3
m = 3
predicted_answer, documents, _memory = system.query(
    question,
    mode=qa_mode,
    max_iter=max_iter,
    retriever_args=retriever_args,
    m=3
)

print("Answer:", predicted_answer)
print("Documents:")
for pmid, data in documents.items():
    print(f"PMID: {pmid}")
    print(f"Passage ID: {data['passage_id']}")
    print(f"Text: {data['text']}")
    print("Facts:")
    for fact in data["facts"]:
        print(f"- {fact}")
    print()
```

### Advanced options at inference

When loading the graph at inference, you can specie a list of ontology nodes to discard
```python
system.retriever.load_graphs(force_reload=True, masks_ontology_ids=["Species|9606"])
```

You can also check the cost
```python
from biotopg.utils.llm import get_cost
cost = get_cost(system.lm)
```
