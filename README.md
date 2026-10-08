# AIR: budgeted molecular screening experiments

This repository contains the experiment implementation and compact numerical records for **AIR** (audit-informed candidate re-entry) and the separately evaluated **AIR-E** acquisition rule. It is a reproduction package, not a copy of the large raw datasets, model checkpoints, or external authors' repositories.

## What is included

| Directory | Contents |
| --- | --- |
| `src/airvs/` | Dataset registry, molecular standardization, budget accounting, and leakage-safe evaluator. |
| `scripts/` | LIT-PCBA preprocessing; AIR and AIR-E experiments; matched classical and author-code baseline adapters; TDC-HTS pool preparation. |
| `tests/` | Unit tests for budget accounting, sampling, acquisition, and guards. |
| `results/source_tables/` | Small, frozen five-seed tables and the 55 AIR-E query trajectories/manifests required to check the reported comparisons. |

The result tables are **our reruns**. The GLARE, AC-Active, TcsAL and ChemScreener results are not copied from their papers. Their implementation/protocol labels are retained in the CSVs. External-route `X*` controls in the original AIR experiment are common-protocol implementations, **not** official author-code reruns; the author-code and matched-pool analyses are reported separately.

The released CSVs preserve every measurement, assay/method/seed row, and row order. A few machine-specific metadata columns (run-file paths, physical GPU IDs, timestamps, and CPU/thread settings) have been removed from six tables. No numerical result was changed by this privacy filter.

## Two non-interchangeable experiment protocols

| Experiment | Data | Initial queries | Total queries | Screened/expensive pool | Rounds | Seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original AIR gate | 11 LIT-PCBA assays | 64 | 192 | 192 | 4 | 5 per assay |
| Matched AIR-E transfer | 11 LIT-PCBA assays | 64 | 192 | shared author-code pool | batches of 64 | 5 selected per assay |
| TDC-HTS development | AID 1798, 463087, 485290 | 64 | 384 | shared screened pool | batches of 64 | 5 per AID |

AIR-E's penalties were selected in TDC-HTS development and then held fixed for LIT-PCBA transfer. Do not merge these three leaderboards or attribute AIR-E results to the original AIR gate.

## Quick verification of the released numerical records

Requires only Python 3.11 or newer and the Python standard library:

```bash
python scripts/verify_published_results.py
```

This read-only check validates the five-seed vectors, reported means and sample SDs, the 55 AIR-E trajectory/query hashes and budgets, and the matched LIT-PCBA/TDC macro summaries. It does not rerun model training.

## Rebuild the LIT-PCBA inputs

Use the [official LIT-PCBA full-data download](https://drugdesign.unistra.fr/LIT-PCBA/) and place its archive at `data/raw/lit_pcba/LIT-PCBA_full.tar.gz`. Extract it so that `data/raw/lit_pcba/extracted/LIT-PCBA_full/<assay>/{actives,inactives}.smi` exists. The 11 main assays are ADRB2, ALDH1, FEN1, GBA, IDH1, KAT2A, MAPK1, MTORC1, OPRK1, PKM2 and VDR. The preprocessing records source checksums and deterministic canonical molecule IDs.

On Linux with Python 3.11:

```bash
python3.11 -m venv envs/air311
envs/air311/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-core.txt
export PYTHONPATH="$PWD/src:$PWD/scripts"
envs/air311/bin/python scripts/build_lit_pcba.py --help
envs/air311/bin/python scripts/build_lit_pcba.py
envs/air311/bin/python scripts/precompute_lit_fingerprints.py --workers 4
envs/air311/bin/python -m pytest -q
```

The experiment scripts use the project-local `envs/air311/bin/{python,chemprop}`. The package and `pytest` commands may also be run in another environment if `PYTHONPATH` is set. Large fingerprint caches, raw labels, and checkpoints are deliberately ignored by Git.

For an original single-task AIR gate run, first inspect available devices and use the built-in preflight/guard checks. The method identifier used in the experimental records is `AIR_p1_legacy_unc_neighbor_gate`:

```bash
envs/air311/bin/python scripts/run_e10_dual_head_pilot.py --self-test-guards
envs/air311/bin/python scripts/run_e10_dual_head_pilot.py --assays ALDH1 --seeds 47001 --methods AIR_p1_legacy_unc_neighbor_gate --prefix example_air --accelerator cpu --cpu-threads 4 --epochs 2 --timeout-sec 1800
```

Do not launch batch GPU jobs merely because a device appears idle. The scripts include device validation and resume flags; validate one task, timeout and checkpoint recovery before parallel execution.

## Matched-pool and external-baseline setup

The following external repositories are **not vendored**. Clone them under `external_baselines/official_code/repos/` at the recorded commits, then install their own documented environments and licenses before running their adapters:

| Directory | Upstream | Recorded commit |
| --- | --- | --- |
| `glare` | [biomed-AI/GLARE](https://github.com/biomed-AI/GLARE) | `fde2859ae03de1de1647c4720f6ed672bf524034` |
| `ac_active` | [wnsgk/AC-Active](https://github.com/wnsgk/AC-Active) | `71bbc315affee9ff3622f3d1fbafbb86380d0822` |
| `chemscreener` | [Novartis/ChemScreener](https://github.com/Novartis/ChemScreener) | `b439341c9801e64d49934baec4288659d70f5d51` |
| `traversing_chem_space` | [molML/traversing_chem_space](https://github.com/molML/traversing_chem_space) | `36a1ed85ddf607e3a0e080adb6097add61c8c235` |

Run `scripts/prepare_glare_acactive_data.py --help` to build the common screened-pool layout. Its `--screen-size 100000`, `--test-size 20000`, and `--split-seed 42` settings reproduce the recorded pool preparation. The matched-pool loaders in `run_e15_matched100k.py` require both author-code `screen.csv` files to hash identically and the recorded official run CSVs in `tables/`. Copies of those small run CSVs are in `results/source_tables/`; copy the relevant `official_glare_*` and `official_acactive_*` files into `tables/` when reconstructing that audit. Keep their filenames unchanged. The complete matched results are in `e96_matched_lit_final_status.csv` and `e98_air_e_transfer/`.

The CPU-only AIR-E transfer is `scripts/e98_lit_air_e.py`. It requires `CUDA_VISIBLE_DEVICES=-1` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`, with a verified matched pool and one of its five collision-free selected seeds. Its `--preflight-only`, `--inject crash|sleep`, and checkpoints permit a single-task control test before any batch run.

## TDC-HTS inputs

The three development tasks come from [Therapeutics Data Commons HTS](https://tdcommons.ai/single_pred_tasks/hts/) and its [Harvard Dataverse data host](https://doi.org/10.7910/DVN/21LKWG). `scripts/e67_tdc_dev_audit.py` expects the original tab-separated files with `Drug_ID`, `Drug`, and `Y` columns in `data/raw/tdc_hts_e64/`:

| AID | Filename | Dataverse file ID | MD5 |
| --- | --- | ---: | --- |
| 1798 | `m1_muscarinic_receptor_agonists_butkiewicz.tab` | 6894443 | `fdcfa9675d671569067fb1ce32d70ee1` |
| 463087 | `cav3_t-type_calcium_channels_butkiewicz.tab` | 6894445 | `acb3a8dff051e57844b7a2ea76359ec3` |
| 485290 | `tyrosyl-dna_phosphodiesterase_butkiewicz.tab` | 6894440 | `e9ecb1cf4201979320361a0d7f93dea5` |

Each file is available from `https://dataverse.harvard.edu/api/access/datafile/<file ID>`. Verify the MD5 before use. Then run `e67_tdc_dev_audit.py`, `e69_tdc_freeze_dev_pool.py`, and `e78_author_starts.py` in that order for each AID. The official graph preparation (`e70_tdc_official_graphs.py`) requires the external AC-Active/GLARE code. The AIR-E acquisition implementation is `e74_tdc_local_risk_candidates.py`, with the transferred variant fixed at negative-neighbor penalty 0.15 and batch-diversity penalty 0.10.

## Scope and provenance

The compact CSVs can reproduce the reported numerical tables without downloading gigabytes of inputs. Full model reruns additionally require the public benchmark downloads, the author-code checkouts, their environments, and sufficient compute. Nothing in this repository is a biological wet-lab validation. Raw benchmark archives, molecular caches, trained models, temporary jobs, private host configuration, and machine-specific logs are intentionally excluded.
