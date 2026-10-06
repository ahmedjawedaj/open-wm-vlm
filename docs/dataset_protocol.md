# Dataset protocol

This document specifies the implemented generator version 2. It describes an independent reconstruction of [WM-VLM](https://arxiv.org/abs/2609.34826v1), not the original benchmark.

## Shapes and split assignment

Connected lattice shapes are enumerated up to proper rotations, preserving chirality. The shape's rotation-and-reflection canonical form and seed determine its split bucket through SHA-256. Mirror-related classes therefore stay together.

Both default datasets train on four-, five- and six-cell shapes. OOD uses seven-cell shapes. The 3D C-ID pool holds out 25 percent of shape classes by hash threshold. Reference and query shapes both come from the sample's pool. ID uses seen shape classes with new symbolic questions. Query-pose/net-rotation configurations can recur with different references, unlike a fully disjoint configuration-pool protocol.

## Question construction

Each question contains one reference shape before and after the shared net rotation, one query, and four options. Reference shapes are symbolically asymmetric. Query shapes must have at least four distinct rotated poses. The reference cell count differs from the query count where a usable alternative exists.

Training and ordinary tests sample one or two actions. A 2D action is a counterclockwise in-plane turn of 90, 180 or 270 degrees. A 3D action uses the right-hand rule about X, Y or Z. Consecutive 3D actions use different axes. Identity net rotations and unchanged query answers are rejected. These constraints make action/step sampling nonuniform after rejection.

The correct option is the final rotated query. Distractors are distinct other poses of that query. Labels are shuffled deterministically. Options with identical rendered pixels are rejected. Exact symbolic states are stored after every action, including in 2D.

## Determinism and duplicate policy

`DetRng` implements SHA-256-seeded SplitMix64. Generation uses exact integer geometry and deterministic ordering. A record regenerates from its config, split, index and variant. The builder tries variants in fixed index order and shares a duplicate set across splits. Sequential and multiprocessing output agree.

A duplicate question is defined by symbolic reference pairs plus query. Hidden traces and option permutations do not distinguish repeated questions. This definition does not ensure pixel uniqueness when different shapes render identically.

The manifest records generator version, config, config hash, Pillow version, per-split hashes and the combined hash. Hashes cover UTF-8 JSONL bytes. They do not cover rendered image bytes. Audits verify provenance, counts, sample order/identity, pool membership, duplicates and optional exact regeneration.

## Rendering and known deviations

Images are 532 × 532 RGB. The 2D board is seven by seven. The 3D renderer uses a fixed isometric camera, integer projection and fixed cell scale. Cubes can be fully occluded, so symbolic answer uniqueness does not prove visual identifiability. See the [dataset card](dataset_card.md).

The target token grid is 19 × 19 under compatible Qwen preprocessing. Actual processor settings and integration have not been validated. Inputs are separate numbered images. The paper's examples use a composite question image, final-only 2D states and repeated-axis 3D quarter turns. This implementation intentionally records its differing behavior rather than claiming exact reproduction.

`depth3_test` supplies three-action supervision traces. A before/after analogy determines a net rotation but not a unique action decomposition or visible trace length. This split alone does not establish reasoning-depth generalization.
