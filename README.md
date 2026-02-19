# triton-kernels

Standalone `triton_kernels` Python package, extracted from the [triton-lang/triton](https://github.com/triton-lang/triton) repository.

## Origin

This package is sourced from [`python/triton_kernels/`](https://github.com/ROCm/triton/tree/release/internal/3.5.x/python/triton_kernels) in the [ROCm/triton](https://github.com/ROCm/triton) fork, branch `release/internal/3.5.x` (commit `443d22e`).

The `triton_kernels` package provides a collection of high-performance Triton kernels for common deep-learning operations.

## Contents

| Module | Description |
|--------|-------------|
| `triton_kernels.matmul_ogs` | Fused matmul with output-group-scatter (OGS) |
| `triton_kernels.routing` | MoE expert routing and token dispatch |
| `triton_kernels.swiglu` | Fused SwiGLU activation |
| `triton_kernels.topk` | Top-K selection (forward + backward) |
| `triton_kernels.compaction` | Masked compaction |
| `triton_kernels.numerics` | Numeric format utilities (MXFP, Flexpoint) |
| `triton_kernels.tensor` | Tensor layout descriptors (Hopper, Blackwell, CDNA4) |
| `triton_kernels.specialize` | Kernel specialization helpers |
| `triton_kernels.target_info` | GPU target information |

### Benchmarks and Tests

- `bench/` — Benchmarking scripts (MLP, roofline, distributed)
- `tests/` — Pytest test suite

## Installation

```bash
pip install -e .
```

Requires: `triton >= 3.5`, `torch`, `numpy`

## License

MIT License — see [LICENSE](LICENSE). Original copyright belongs to the Triton authors (Philippe Tillet, OpenAI).
