# K-Means Clustering Implementation

## Implementation Details

### Main Script: `kmeans-emb.py`

The implementation performs K-Means clustering across multiple K values to enable comparison and selection of optimal cluster counts.

**Key Features:**
- Loads node embeddings (Node2Vec or Role2Vec)
- Runs K-Means with multiple K values in a single execution
- Saves all clustering results in a single CSV file

**Algorithm Configuration:**
```python
K_VALUES = [10, 15, 20, 30, 40, 50, 65, 70, 75, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400]
random_state = 42
n_init = 10  # Number of initializations
```

## Input Format

**Embedding CSV File:**
- Must contain an `account_id` column (node identifier)
- Remaining columns are embedding dimensions (e.g., z1, z2, ..., z128)


## Output Format

**Output CSV:**
- Original embedding file with added cluster assignment columns
- One column per K value: `kmeans_{k}`
- Column values are cluster labels (0 to k-1)

**Example Structure:**
```csv
account_id,z1,z2,...,z128,kmeans_10,kmeans_15,kmeans_20,...
12345,0.123,-0.456,...,0.234,3,7,12,...
67890,0.567,0.123,...,0.678,3,14,19,...
```

## Usage Example

### Basic Usage

- Navigate to K-Means directory

```bash
python kmeans-emb.py
```


## Dependencies

**Required Python Packages:**`pandas`, `scikit-learn`, `numpy`

**Installation:**
```bash
pip install pandas scikit-learn numpy
```

 K-Means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html