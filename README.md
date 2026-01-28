# Riemannian-Embedded Transformer

A PyTorch implementation of a Transformer architecture that transitions internal embedding dynamics from a flat Euclidean space to a **continuous Riemannian manifold**. This project parameterizes the local geometry via a learned **metric tensor ($g_{\mu\nu}$)** to capture intrinsic data hierarchies and non-linear dependencies.



## Abstract
Standard Transformer architectures operate under the assumption of a flat Euclidean geometry, often failing to efficiently represent complex hierarchical or cyclical structures. This project reformulates the self-attention mechanism as a **metric contraction** within the **cotangent bundle** of a manifold. By treating token interactions as **geodesic flows**, the model imposes a geometric inductive bias that enhances representational capacity while maintaining parameter efficiency.

## Technical Architecture

### 1. Dynamic Metric Generation
Unlike static hyperbolic embeddings, this model utilizes a neural network to generate a local metric tensor $g_{\mu\nu}$ for each input token. 
* **Symmetry & Positivity:** The metric is enforced as symmetric positive-definite via Cholesky decomposition ($g = LL^T$).
* **Local Curvature:** The geometry is learned dynamically, allowing the manifold to "warp" in response to specific semantic features.

### 2. Riemannian Attention Mechanism
The traditional dot-product attention $QK^T$ is replaced by the contraction:
$$\text{Score} = Q^\mu g_{\mu\nu} K^\nu$$
This ensures that attention weights are a function of the **local manifold curvature**, effectively acting as a geometric gatekeeper for information flow.

### 3. Geodesic Propagation
Information update steps are modeled as trajectories along **geodesic paths**, utilizing **parallel transport** to maintain vector orientation across successive layers.



## Technical Stack
* **Framework:** PyTorch
* **Mathematics:** Differential Geometry, Riemannian Manifolds, Tensor Calculus
* **Core Concepts:** Cotangent Bundles, Metric Contractions, Geodesic Flow

## Project Status (2026)
* **Core Logic:** Implemented metric generator and contraction layers.
* **Baseline:** Benchmarking against standard Euclidean Transformers on hierarchical datasets to measure parameter efficiency and loss convergence.
