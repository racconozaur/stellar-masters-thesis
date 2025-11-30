# Behavior-Based Address Clustering on Stellar Blockchain

## Project Overview

This repository contains the implementation code for a masters thesis analyzing community detection and clustering algorithms on the Stellar blockchain network. The project evaluates various graph based and embedding based methods for identifying communities within Stellars transaction and trustlines networks.

## Contents

- **Python scripts**: Data preprocessing, algorithm implementation, and evaluation modules
- **Jupyter notebooks**: For analysis, results, visualization

## Purpose

This codebase implements and evaluates multiple community detection approaches on Stellar network data, comparing unsupervised clustering methods (K-Means, DBSCAN), graph-based community detection algorithms (Louvain, LPA, SSLPA, Spectral Clustering). The research aims to identify optimal methods for detecting meaningful communities in blockchain transaction networks.

## Main Directories

- **data/**: Raw network data and preprocessing scripts for Largest Connected Component (LCC) extraction
- **labled-data/**: Scripts for fetching and normalizing labels from Stellar Expert
- **clustering/**: Implementation of K-Means and DBSCAN algorithms with Node2Vec and Role2Vec embeddings
- **community detection/**: Graph-based algorithms Louvain, Spectral Clustering, Label Propagation, Semi-Supervised Label Propagation

