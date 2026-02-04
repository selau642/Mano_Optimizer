
#Manu Optimizer
This is a reimplimentation of the optimizer invented by Yufei Gu and Zeke Xie

```
@misc{jordan2024muon,
  author       = {Yufei Gu and Zeke Xie},
  title        = {Manu: Restriking Manifold Optimization for LLM Trainingj},
  year         = {2026},
  url          = {https://arxiv.org/pdf/2601.23000/}
}
```

# Key properties
Normalizes weight matrix via Sinkhorn-Knopp algorithm to ensure weight matrix are doubly stochastic matrix. 
both column space and row space vectors normalized to 1.