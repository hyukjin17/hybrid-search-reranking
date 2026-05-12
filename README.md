# Hybrid Search & Intent-Aware Reranking

<img width="1102" height="632" alt="Search Demo" src="https://github.com/user-attachments/assets/cc139870-2010-441c-b3ed-4643ff6c0b91" />

Hybrid e-commerce search pipeline that combines **lexical matching (BM25)** and **semantic understanding (Two-Tower BERT)** with an **Intent-Aware Neural Reranker**

Built on the Amazon ESCI (Shopping Queries) dataset, this project tackles the "Homogeneity Trap" in e-commerce search by dynamically balancing text relevance, semantic intent, and product popularity based on the complexity of the user's query

---

## The Problem: Vague vs. Complex Queries
Standard search engines often struggle to adapt to user intent:
* **Vague Queries (e.g., *"shoes"*):** Standard engines often return random exact-text matches. Users actually want to see highly popular, well-reviewed items from diverse sub-categories (Exploration).
* **Complex Queries (e.g., *"red nike running shoes under $100"*):** Standard engines heavily weight popularity, returning best-selling items that ignore the strict constraints (Precision).

**Our Solution:** A hybrid pipeline that routes candidates through a deep neural network trained via Pairwise Ranking, applying dynamic weights to price, popularity, and text-match based on query specificity (IDF).

## Best of Both Worlds

| Query Type   | Behavior                          |
| ------------ | --------------------------------- |
| Exact        | BM25 dominates                    |
| Conceptual   | Semantic dominates                |
| Ambiguous    | Inconsistent ranking behavior     |

- **Lexical search (BM25)** is fast but struggles with synonyms and semantic intent.
- **Semantic neural retrieval** captures meaning but suffers from domain gap and inconsistency on exact-match queries.

---

## Core Architecture

### Stage 1: Hybrid Retrieval 
* **Lexical Retrieval (BM25):** Handles exact SKU matches, brand names, and highly specific vocabulary.
* **Semantic Retrieval (Two-Tower BERT):** Captures the latent meaning of queries using FAISS for high-speed vector similarity search.
* **Merge & Deduplicate:** Fetches the Top-K candidates from both indices to ensure high Recall (and removes duplicate results).

### Stage 2: Intent-Aware Neural Reranker
* **Architecture:** A deep Feed-Forward Network trained to predict the final utility of a query-product pair.
* **Pairwise Learning:** Utilizes `MarginRankingLoss`, `LayerNorm`, and `LeakyReLU` to evaluate pairs of items (Item A vs. Item B). By learning relative ranking margins instead of pointwise global averages, the network avoids flatlining on heavily imbalanced tabular data.
* **Engineered Features (14+ Dimensions):**
  * *Relevance:* BM25 Score, Semantic Cosine Similarity, Exact Word Overlap.
  * *Query Complexity:* IDF (Inverse Document Frequency) Mean, Query Length.
  * *Implicit Popularity & Trust:* Brand Frequency, Product Frequency, Bullet Point Count.
  * *Business Logic (from ESCI-S):* Price Percentile, Review Score, Review Count, Price Intent Keywords.

```
TRAINING
ESCI / ESCI-S Data ──► BM25 + Two-Tower Indexing ──► Sparse & Dense Index files
                                                              │
                                                              ▼
                                              Reranker (Pairwise MLP)
                                                              │
                                                              ▼
                                       Weights (.pth) + Normalization Stats (.json)
 
INFERENCE
User Query ──► BM25 + Two-Tower Lookup ──► Merge + Dedupe ──► Reranker + MMR ──► Top-20
              [RETRIEVAL]                  [FUSION]          [RANKING]
                  │                            │                  │
              2.6M indexed              150-300 candidates   20 final results
```

---
 
## Dataset
 
We use the **Amazon ESCI dataset** combined with **ESCI-S** for richer product signals.
 
**ESCI (base)**
- 130K+ queries, 2.6M+ human-labeled judgements, 3 languages
- ~20 labeled items per query (median 16)
- Four-way relevance labels: **E**xact, **S**ubstitute, **C**omplement, **I**rrelevant
- Native train/test split; we further split training into 85/15 train/validation
**ESCI-S (enrichment)**
- Star ratings, review counts, price, and category hierarchy
- Covers 1.66M products
- Fills the gap left by ESCI's text-only labels — provides the behavioral and numerical signals real ranking systems rely on
Together they give the pipeline both semantic richness and real-world product signals.
 
---

## Custom Evaluation: Adaptive-NDCG (A-NDCG)
Standard NDCG strictly penalizes non-exact matches. We developed **Adaptive-NDCG** to evaluate the model based on actual user intent:
* For **Vague Queries (Low IDF)**, highly-rated "Substitute" items receive a boosted Gain score, as the user intent is exploratory window-shopping.
* For **Specific Queries (High IDF)**, strict penalties are applied to items that violate explicit constraints (e.g., missing a brand or price limit), preserving precision.

---

## Project Structure

```text
.
├── retrieval/          # First-stage retrieval logic
│   ├── bm25.py         # BM25 lexical scoring and FAISS index building
│   └── two_tower.py    # Two-tower semantic scoring and index building
├── reranking/          # Second-stage neural reranking
│   ├── model.py        # Deep Pairwise Reranker architecture (LayerNorm + LeakyReLU)
│   └── features.py     # Feature extraction + Pairwise ESCI Dataset loaders
├── evaluation/         # Metrics and evaluation
│   ├── metrics.py      # Standard NDCG@10, Recall@K, and Custom A-NDCG
│   ├── evaluate_retrieval.py
│   └── evaluate_reranker.py
├── analysis/           # Query complexity and intent analysis
│   ├── concept_entropy.py
│   └── idf_setup.py    # IDF specificity calculations
├── scripts/            # Executable entry points
│   ├── run_pipeline.py             # End-to-end pipeline execution
│   ├── build_indices.py            # Pre-compute BM25 + FAISS embeddings
│   ├── generate_bm25_scores.py     # Batch generate Train/Test BM25 scores
│   ├── generate_two_tower_scores.py# Batch generate Train/Test Semantic scores
│   ├── train_two_tower.py          # Fine-tune the HuggingFace BERT encoder
│   └── train_reranker.py           # Train the Deep Pairwise Reranker
├── tests/              # Unit tests for retrieval and reranking logic
├── models/             # Saved .pth weights (Git-ignored)
├── output/             # Generated CSV scores and predictions (Git-ignored)
├── esci-data/          # Amazon ESCI & ESCI-S Parquet datasets (Git-ignored)
└── config.py           # Global paths, split controls, and hyperparameters
```

## How to run

Everything is run from the project root. Follow these steps in order:

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Fine-tune the Two-Tower model** (only needed once)
```bash
python scripts/train_two_tower.py
```
This trains the semantic encoder on ESCI data and saves weights to `models/two_tower_finetuned/`.

**Step 3 — Generate retrieval scores** (only needed once, or when data changes)
```bash
python scripts/generate_bm25_scores.py
python scripts/generate_two_tower_scores.py
```
These produce `output/bm25_scores_{split}.csv` and `output/two_tower_scores_{split}.csv` for train and test.

**Step 4 — Build search indices** (only needed once)
```bash
python scripts/build_indices.py
```
Pre-computes BM25 and FAISS indices for fast retrieval at search time.

**Step 5 — Train the reranker**
```bash
python scripts/train_reranker.py
```
Trains the neural reranker using the scores from Step 3. Saves the best model to `output/best_esci_reranker.pth`.

**Step 6 — Evaluate**
```bash
python evaluation/evaluate_retrieval.py
python evaluation/evaluate_reranker.py
```
Prints NDCG@10 and Recall@K for the retrieval baselines and the reranker.

**Step 7 — Run the full pipeline** (optional)
```bash
python scripts/run_pipeline.py
```

**Try an interactive search** (requires Steps 2-5 completed)
```bash
python tests/test_custom_search.py
```
Type a query and see ranked results from the full pipeline.

## Tests

Run from the project root:

```bash
python3 -m tests.test_two_tower
```

The test suite validates the two-tower retrieval logic against a small synthetic dataset without requiring real data or GPU. It checks:

- **Output shape** — result contains exactly the three expected columns (`query_id`, `item_id`, `two_tower_score`)
- **No empty results** — the function returns at least one row
- **Score normalization** — all scores are in [0, 1]
- **Query coverage** — every query in the input appears in the output
- **Item coverage** — every item in the input appears in the output
- **Semantic relevance** — the model ranks semantically relevant items higher than irrelevant ones (e.g. "running shoes" scores higher than "coffee maker" for a running shoes query)
- **No duplicates** — each item appears only once per query
- **Input validation** — passing a dataframe with missing required columns raises a `ValueError`
