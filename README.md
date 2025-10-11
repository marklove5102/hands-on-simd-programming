![Intel Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Intel_logo_%282006-2020%29.svg/200px-Intel_logo_%282006-2020%29.svg.png)

# Hands-on SIMD Programming with C++

From “what is SIMD?” to “how do I speed up real workloads?”, this repository walks through reproducible microbenchmarks, AVX2 intrinsics, and even a transformer-style attention block.

![Intel ISA Families and Features](./assets/intel_isa_families.jpeg)

![SIMD Speedups](artifacts/benchmark_speedups.png)

![Attention Breakdown](artifacts/attention_speedups.png)

## Quick Start

```bash
./runme.sh
# optional: rerun the plotting script manually if you tweak the CSVs
python scripts/plot_results.py
```

`runme.sh` already refreshes both plots automatically; the explicit commands above are only needed if you want to regenerate from modified data. All generated CSV/PNG artifacts live under [`artifacts/`](artifacts) so the project root stays tidy.

## What’s Inside?

| Module | Highlights | Use Cases / Benchmarks |
| --- | --- | --- |
| **01_Basics** | Loads, alignment, data initialisation, intrinsics setup | `01_importing_simd`, `04_loading_data` |
| **02_Computations** | Vector arithmetic, FMA, structure-of-arrays vs. array-of-structures dot product | `01_simple_maths`, `02_dot_product` |
| **03_Examples** | Masked control flow, quadratic solver, image operators, transformer attention block | `01_conditional_code`, `04_image_processing`, `05_mha_block` |

- Every example ships with scalar **vs.** SIMD implementations and an embedded benchmark so you can quantify the payoff.
- `03_Examples/05_mha_block` packages RMSNorm + multi-head attention (MHA) + feed-forward into a single transformer block and emits a stage-by-stage CSV for deeper analysis.

## How to Read the Figures

1. **SIMD Speedups** – the six canonical scenarios highlighted in this tutorial. You can immediately see alignment effects, arithmetic speedups, AoS→SoA wins, mask-driven branching, batched equation solving, and image processing kernels ranging from 1.3× to 40× acceleration.
2. **Attention Breakdown** – reserved for the transformer block:
   - Left: per-component speedups (RMSNorm, QKV projections, FFN, etc.).
   - Centre: end-to-end latency comparison (≈2.8× faster with SIMD).
   - Right: each component’s contribution to the overall time saved.

## Key Takeaways

- **Memory layout strategy** – transpose and SoA conversions keep SIMD loads contiguous.
- **Intrinsic choices** – `_mm256_fmadd_ps`, `_mm256_max_ps`, `_mm256_maskload_ps`, and friends are demonstrated in real contexts.
- **Accuracy checks** – SIMD outputs are always compared to scalar references (typical max error ≈ 2e-5 in the attention block).
- **Automation** – `runme.sh` rebuilds every sample, records `benchmark_results.csv`, and the plotting scripts turn that data into publication-ready figures.

## License

MIT
