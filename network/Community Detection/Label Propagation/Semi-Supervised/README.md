## Semi-Supervised Label Propagation


### Implementation: 

**Main Function:**
```python
python ss-lpa-igr.py
```



**Configuration:**
```python
LABELS = "labels_dir"
weights = 'weight'
```

### Input Format

**Graph Pickle File:**
Same as unsupervised (NetworkX Graph object)



### Output Format (Semi-Supervised)

**Label Assignment CSV:** `{prefix}_sslpa_labels.csv`
```csv
node,label
12345,exchange
67890,exchange
11223,anchor
44556,market_maker
99999,CLUSTER_5
```

**Community Mapping CSV:** `{prefix}_lpa_communities.csv`
```csv
node,community
12345,0
67890,0
11223,1
44556,2
```


## Related Files
- **Seed Labels:** Ground truth labels from Stellar Expert for semi-supervised



## Dependencies

```bash
pip install networkx pandas igraph scikit-learn
```

IGraph LPA: https://igraph.org/r/html/1.2.5/cluster_label_prop.html 


