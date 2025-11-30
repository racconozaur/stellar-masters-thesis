# Role2Vec Graph Embedding Implementation

## Implementation Details


### 1) Main Script: `prepare-data.py`

This is a preprocessing script that converts NetworkX graphs into the format required by the Role2Vec tool. 

**Key Features:**
- Loads NetworkX graph pickle files
- Converts node IDs to integer indices (required by Role2Vec)
- Exports edge list in CSV format
- Creates node mapping file to restore original IDs
- Generates command templates for running Role2Vec
- Processes both transaction and trustline networks

### 2)  Role2Vec Implementation

You need to clone Role2Vec separately

Repository: https://github.com/benedekrozemberczki/role2vec


### 3)  Secondary Script: `map-embeddings-back.py`

Converts Role2Vec output (integer indices) back to original account IDs using the mapping files.


---
## Execution Structure
1. Data Preparation (`prepare-data.py`)
1. Role2Vec Execution (GitHub Tool)
1. Mapping Back (Optional)

**Final Output:**
```csv
account_id,z1,z2,z3,...,z128
12345,0.123,-0.456,0.789,...,0.234
67890,0.567,0.123,-0.345,...,0.678
...
```

## Key Parameters 

### Structural Feature Parameters (in Role2Vec command)

#### --features (str, default="wl")
- **Purpose:** Type of structural features to extract
- **Options:** "wl" (Weisfeiler-Lehman)

#### --labeling-iterations (int, default=2)
- **Purpose:** Number of WL refinement iterations
- **Effect:**
  1. Higher: Captures larger neighborhood patterns, more computational power
  1. Lower: Only immediate local structure

#### --log-base (int, default=2)
- **Purpose:** Base for logarithmic binning of features



### Random Walk Parameters

#### --sampling (str, default="second")
- **Purpose:** Walk sampling strategy
- **Options:** "first", "second", "meta"
- **Effect:** "second" is recommended for role-based walks

#### --P (float, default=1.0)
- **Purpose:** Return parameter (similar to Node2Vec)
- **Effect:** Controls backtracking probability


#### --Q (float, default=4.0)
- **Purpose:** In-out parameter (similar to Node2Vec)
- **Effect:**
  1. Higher Q: More structural equivalence focus
  1. Lower Q: More proximity focus

#### --walk-number (int, default=10)
- **Purpose:** Number of walks per node
- **Effect:** More walks = better coverage, slower

#### --walk-length (int, default=80)
- **Purpose:** Steps in each random walk
- **Effect:** Longer walks capture broader structural context

### Embedding Parameters

#### --dimensions (int, default=128)
- **Purpose:** Embedding vector dimensionality
- **Effect:** Higher dimensions capture more structural nuance

#### --epochs (int, default=1)
- **Purpose:** Word2Vec training epochs
- **Effect:** More epochs may improve quality but increase runtime

#### --workers (int, default=8)
- **Purpose:** Number of parallel threads


## Usage Example

### Complete Workflow

- Navigate to Role2Vec directory
- Prepare data (convert graphs to Role2Vec format)

```bash
python prepare-data.py
```

-  Run Role2Vec (GitHub)

```bash

# Iside Role2Vec implementation directory
python role2vec/src/main.py \
  --graph-input  src/tx_lcc/edges.csv \
  --output       src/tx_lcc/role2vec_raw.csv \
  --features wl --labeling-iterations 2 --log-base 2 \
  --sampling second --P 1.0 --Q 4.0 \
  --dimensions 128 --window-size 5 --walk-number 10 --walk-length 80 \
  --epochs 1 --workers 8 --seed 42

```

- Map embeddings back to original account IDs

```
python map-embeddings-back.py
```

## Dependencies

**Python Packages (for prepare-data.py):**, `networkx`, `pandas`, `pickle`

**Installation:**
```bash
pip install networkx pandas
```

**External Tool (Role2Vec):**
- Repository: https://github.com/benedekrozemberczki/role2vec

**Installation:**
```bash
# Clone Role2Vec repository
git clone https://github.com/benedekrozemberczki/role2vec.git
cd role2vec

# Install dependencies
pip install -r requirements.txt
```

