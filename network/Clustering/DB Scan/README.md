# DBSCAN Clustering Implementation

## Implementation Details

### Main Script: `dbscan-final.py`

The implementation uses an parameter selection strategy based on k-nearest neighbor distances to automatically determine appropriate epsilon (eps) values for different min_samples settings.


**Configuration:**
```python
MIN_SAMPLES_LIST = [5, 10, 15]      # Core point threshold
PERCENTILES = [70, 80, 85, 90, 95]  # k-NN distance percentiles for eps
PCA_COMPONENTS = 64                  # Dimensionality reduction
METRIC = "cosine"                    # Distance metric
ALGORITHM = "brute"                  # Exact distance computation
```

## Input Format

**Embedding CSV File:**
- Must contain an `account_id` column (node identifier)
- Remaining columns are embedding dimensions (e.g., z1, z2, ..., z128)

## Output Format

**Output CSV:**
- Original account IDs with multiple DBSCAN clustering results
- One column per parameter combination
- Column naming: `dbscan_ms{min_samples}_p{percentile}_eps_{epsilon_value}`
- Cluster labels: 0, 1, 2, ... (clusters), -1 (noise/outliers)


**Console Output:**
```
k=5 seeds: {70: 0.1234, 80: 0.1567, 85: 0.1789, 90: 0.2012, 95: 0.2456}
[ms=5 p=70 eps=0.1234] clusters=25 | noise=1234
[ms=5 p=80 eps=0.1567] clusters=18 | noise=2345
```

## Key Parameters

### eps (float) - Epsilon Neighborhood Radius
- **Purpose:** Maximum distance between two points to be considered neighbors
- **Effect:**
  - Lower eps: More noise, smaller,more clusters
  - Higher eps: Less noise, larger,fewer clusters


### min_samples (int)
- **Purpose:** Minimum number of points required to form a dense region (core point)
- **Effect:**
  - Lower values: More clusters, less noise, more sensitive to density variations
  - Higher values: Fewer clusters, more noise, stricter density requirements

### PERCENTILES (List[int])
- **Purpose:** Determines eps from k-NN distance distribution


### PCA_COMPONENTS (int, default=64)
- **Purpose:** Reduce dimensionality before clustering
- **Effect:**
  - Speeds up distance computations
- **Trade-off:** Performance vs information

### metric (str, default="cosine")
- **Purpose:** Distance/similarity measure
- **Options:**
  - `"cosine"`: Angle-based similarity (recommended for embeddings)
  - `"euclidean"`: Straight-line distance
  - `"manhattan"`: City-block distance
- **Note:** Cosine distance works well for high-dimensional embeddings

## Usage Example

### Basic Usage

- Navigate to DBSCAN directory

```bash
python dbscan-final.py
```




## Dependencies

**Required Python Packages:**`pandas`, `scikit-learn`, `numpy`

**Installation:**
```bash
pip install pandas scikit-learn numpy
```

DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html 