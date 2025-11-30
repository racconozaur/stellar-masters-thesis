## 1. Transaction Network LCC

### Script: `create-transaction-lcc.py`

### Usage

```bash
python create-transaction-lcc.py
```

### Purpose

Extracts the largest connected component from the undirected, weighted transaction graph of the Stellar network. This creates a clean, fully connected network suitable for community detection analysis.


### Implementation Details

1. **Load Full Graph**: Reads the complete transaction network from a pickle file
2. **Find Connected Components**: Uses NetworkX's `connected_components()` to identify all connected components
3. **Extract LCC**: Selects the largest component by node count
4. **Create Subgraph**: Generates a copy of the LCC subgraph preserving all node and edge attributes
5. **Save Output**: Serializes the LCC graph to a pickle file

### Input Requirements

**Format**: NetworkX Graph object (pickl)

**Graph Properties**:
- Type: Undirected, weighted graph
- Nodes: Stellar account IDs (integers)
- Node Attributes: Account-specific features
- Edge Attributes:
  - `weight`: Transaction metrics (transaction count or volume)


### Output

**Output File**:


**Format**: NetworkX Graph object

**Contents**:
- All nodes in the largest connected component
- All edges connecting those nodes
- All original node and edge attributes preserved



## Dependencies

**Required Python Packages:**`networkx`, `pickle`

**Installation:**
```bash
pip install networkx 
```

