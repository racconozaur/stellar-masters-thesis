# Node2Vec Graph Embedding Implementation

## Implementation Details


### Main Script:

This implementation generates Node2Vec embeddings of the transaction and trustlines networks Largest Connected Component (LCC).


**Algorithm Configuration:**
```python
EMBED_DIM = 128        # Embedding dimensionality
WALK_LENGTH = 30       # Steps per random walk
NUM_WALKS = 4          # Walks per node
WINDOW = 10            # Word2Vec context window
P = 1.0                # Return parameter (BFS vs DFS)
Q = 2.0                # In-out parameter (local vs global)
SEED = 42              # Random seed
WORKERS = 2            # Parallel threads
```

## Input Format

**Graph Pickle File:**
- NetworkX graph object (undirected, weighted)
- Nodes: Account IDs (integers or strings)
- Edges: Must have `weight` attribute 




## Output Format

**Output CSV:**
- First column: `account_id` (original node identifiers)
- Remaining columns: `z1, z2, ..., z128` (embedding dimensions)
- Each row represents one node's embedding vector



## Usage Example

### Basic Usage

```bash
cd your dir

python node2vec_tx-lcc.py
```


## Dependencies

**Required Python Packages:**`networkx`, `node2vec`, `pandas`, `numpy`

**Installation:**
```bash
pip install networkx node2vec pandas numpy
```

**Note:** The script uses the `node2vec` library from PyPI (https://github.com/eliorc/node2vec), not the original implementation.
