## Unsupervised Label Propagation

### Implementation

**Main Function:**
```
python unsupervised-lpa.py
```

**Default Configuration:**
```python
seed = 42                # Random seed
weight = "weight"        # Edge attribute for weights
```

### Input Format (Unsupervised)

**Graph Pickle File:**
- NetworkX Graph object
- Must have `weight` edge attribute


### Output Format (Unsupervised)

**Community Assignment CSV:** `{prefix}_lpa_communities.csv`
```csv
node,community
12345,0
67890,1
11223,0
44556,2
```



## Dependencies

```bash
pip install networkx pandas
```


asyn_lpa_communities: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.asyn_lpa_communities.html
