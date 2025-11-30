# Louvain Method for Community Detection

## Implementation Details

### Main Script: `louvian-different-resolutions.py`

This implementation runs Louvain community detection with multiple resolution parameters to explore different granularities of community structure in the network



**Algorithm Configuration:**
```python
RESOLUTIONS = [0.5, 0.8, 1.0, 1.2]  # Resolution parameter values
THRESHOLD = 1e-7                     # Convergence threshold
SEED = 42                             # Random seed
WEIGHT = "weight"                     # Edge attribute for weights
```


## Input Format

**Graph Pickle File:**
- NetworkX Graph or DiGraph object
- Must have weighted edges with `weight` attribute
- Can contain isolated nodes (will be removed)


## Output Format

### Per-Resolution Files

**Filename Pattern:** `louvain_result_res{resolution}.csv`

**Structure:**
```csv
account_id,community,resolution
12345,0,0.5
67890,0,0.5
11223,1,0.5
44556,2,0.5
...
```

**Fields:**
- `account_id`: Original node identifier
- `community`: Community ID (0-indexed, sorted by size descending)
- `resolution`: Resolution parameter used


## Key Parameters

### resolution (float, default=1.0)
- **Purpose:** Controls the granularity of detected communities
- **Effect:**
  - Lower values (0.5-0.9): Fewer, larger communities (coarse-grained)
  - 1.0: Standard modularity optimization (default)
  - Higher values (1.1-2.0): More, smaller communities (fine-grained)


### threshold (float, default=1e-7)
- **Purpose:** Convergence criterion for optimization

## Usage Example

### Basic Usage - Transaction Network

- Navigate to transaction directory

```bash
python louvian-different-resolutions.py
```

## Dependencies

**Required Python Packages:**`networkx`, `pandas`, `pickle`

**Installation:**
```bash
pip install networkx pandas
```

Louvain: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html 