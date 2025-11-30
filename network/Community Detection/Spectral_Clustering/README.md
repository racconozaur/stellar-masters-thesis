# Spectral Clustering for Community Detection

## Implementation Details

### Main Script: `spectral-graph.py`

This implementation applies Spectral Clustering to graph data with multiple k values, enabling comparison of different cluster granularities

**Key Features:**
- Tests multiple k (number of clusters) values in a single run
- Builds affinity matrix directly from graph structure
- Uses efficient sparse matrix operations
- Employs LOBPCG eigensolver for large graphs
- Ensures symmetric, positive affinity matrix
- Deterministic results with random seed

**Algorithm Configuration:**
```python
K_LIST = [10, 15, 20, 30, 40, 50, 65, 70, 75, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400]
RANDOM_STATE = 42
N_INIT = 10              # K-Means initializations
EIGEN_SOLVER = "lobpcg"  # Eigensolver (fast for large sparse matrices)
ASSIGN_LABELS = "kmeans" # Cluster assignment method
```

## Input Format

**Graph Pickle File:**
- NetworkX Graph object (undirected, weighted)
- Nodes: Account IDs
- Edges: Must have `weight` attribute


## Output Format

### Main Output File

**Fields:**
- `account_id`: Original node identifier
- `cluster_k{k}`: Cluster assignment for k clusters (0-indexed)
- Multiple k columns enable comparison

**Structure:**
```csv
account_id,cluster_k10,cluster_k15,cluster_k20,cluster_k30,...
12345,3,7,12,25,...
67890,3,14,19,38,...
11223,1,5,8,15,...
```




## Key Parameters

### n_clusters (int) - K Value
- **Purpose:** Number of clusters to identify
- **Effect:** Determines granularity of community structure


### affinity (str, default="precomputed")
- **Purpose:** How to construct the affinity matrix
- **Options:**
  - `"precomputed"`: Use provided affinity matrix (used here)
  - `"rbf"`: Radial basis function kernel
  - `"nearest_neighbors"`: k-NN graph

### assign_labels (str, default="kmeans")
- **Purpose:** Method for assigning clusters from eigenvectors
- **Options:**
  - `"kmeans"`: K-Means on spectral embedding (default)
  - `"discretize"`: Direct discretization (faster, lower quality)


### eigen_solver (str, default="lobpcg")
- **Purpose:** Algorithm for computing eigenvectors
- **Options:**
  - `"lobpcg"`: Locally Optimal Block Preconditioned Conjugate Gradient (fast, sparse)
  - `"arpack"`: ARPACK eigensolver (alternative)
  - `"amg"`: Algebraic Multigrid (very large graphs)


### n_init (int, default=10)
- **Purpose:** Number of K-Means initializations
- **Effect:**
  - Higher: Better final clustering, slower
  - Lower: Faster, may get suboptimal result

## Usage Example

### Basic Usage

- Navigate to Spectral Clustering directory
```bash
python spectral-graph.py
```

## Dependencies

**Required Python Packages:**`networkx`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `pickle`

**Installation:**
```bash
pip install networkx scikit-learn scipy numpy pandas
```

Spectral Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html 

