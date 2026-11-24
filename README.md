# Ramsey multiplicity of K₄: C₈₉ ≤ 0.0301388366

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24517224.svg)](https://doi.org/10.5281/zenodo.24517224)

This repository contains a step graphon (a weighted 2-colouring) whose monochromatic K₄ density is exactly

```
73070278155526479765345578606645339095483962096083909137 / 2424455836135488232949728695296387575646833376892026880000
= 0.0301388365448629…
```

This proves that the Ramsey multiplicity constant of K₄ is at most 0.0301388366.

| Source | Upper bound |
|---|---|
| Parczyk, Pokutta, Spiegel and Szabó (2024) | < 0.030145 |
| Van Seters and Jothimani (2026) | 0.0301388876 |
| Koshta (2026) | 0.0301388857 |
| SK (2026) | 0.0301388794 |
| **this repository** | **0.0301388366** |

The new value is below SK (2026) by 34563360039651519955026999638538156798548911287461 / 808151945378496077649909565098795858548944458964008960000 ≈ 4.28 · 10⁻⁸.

## Witness

`witness/c89_w3_D24.json.gz` uses the format `rational-step-graphon-v1`, the same as SK (2026):

- 7680 blocks of equal weight;
- red probabilities R_ij / 2²⁴.

Its SHA-256 is in `witness/SHA256SUMS`.

## Construction

The construction starts from the SK (2026) step graphon (3840 blocks).

1. Every fractional block-pair entry is optimized individually. SK (2026) instead moved entries together in symmetry classes.
2. One balanced sign refinement splits each block into two, giving 7680 blocks.
3. The refined graphon is optimized further and rounded to multiples of 2⁻²⁴.

## Verification

- `code/exact_struct.py` gives the exact density. It computes the integer numerator over every block modulo 10 primes and combines them by CRT, plus one extra check prime. Output: `results/exact_w3.txt`.
- The same evaluator reproduces the exact SK (2026) value: `results/exact_sk_replay.txt`.
- Two independent float64 checks share no code with the evaluator:
  - `code/dense_float_check.py` checks individual rows (`results/dense_rowcheck_w3.txt`);
  - `code/sampled_check.py` averages 40 random rows (`results/sampled_check_40rows.txt`).
- `code/test_exact_small.py` compares the evaluator with brute force on small random graphons.

To reproduce (Python 3 with numpy; the exact count takes about an hour on 4 cores and about 10 GB of RAM):

```
sha256sum -c witness/SHA256SUMS
cd code
python3 exact_struct.py ../witness/c89_w3_D24.json.gz 40 4
python3 test_exact_small.py
```

## Credits

The construction refines the step graphons of SK (2026), Van Seters and Jothimani (2026), and Parczyk, Pokutta, Spiegel and Szabó (2024):

- J. SK, *Certified step graphon for the Ramsey multiplicity of K₄*, github.com/JeetuSK0808/tao-constants-certificates (2026).
- L. Van Seters and M. Jothimani, *Twelve Clebsch graphs and a new upper bound for the Ramsey multiplicity of K₄*, github.com/lukevs/k4-ramsey (2026).
- O. Parczyk, S. Pokutta, C. Spiegel and T. Szabó, *New Ramsey multiplicity bounds and search heuristics*, Foundations of Computational Mathematics (2024).

The search strategy was conceived by the author and the code developed through many a back-and-forth with Claude.

## Citation

Emirhan Inan, *Ramsey multiplicity of K₄: a step graphon certificate for C₈₉ ≤ 0.0301388366* (2026). https://doi.org/10.5281/zenodo.24517224

## License

CC BY 4.0 (see `LICENSE`).
