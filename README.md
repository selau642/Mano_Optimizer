
# Mano Optimizer
This is a reimplimentation of the optimizer invented by Yufei Gu and Zeke Xie  

```
@misc{
  author       = {Yufei Gu and Zeke Xie},
  title        = {Mano: Restriking Manifold Optimization for LLM Trainingj},
  year         = {2026},
  url          = {https://arxiv.org/pdf/2601.23000/}
}
```

# Key properties
1. This optimizer normalizes weight matrix via Sinkhorn-Knopp algorithm to ensure weight matrix are doubly stochastic matrix.  
2. Both column space and row space vectors in a M x N matrix are normalized to unit norm.  