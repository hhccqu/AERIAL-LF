# Archived F15 Result Card

The manuscript result is tied to one immutable recorded input release:

- 97 physical scenes and 181 command-conditioned routes;
- five scene/group-disjoint folds;
- seeds 20260827, 20260828, and 20260829;
- 15 trained fusion models;
- per-route prediction averaging before metric computation.

## Recorded Metrics

| Subset | Vehicle only ADE/FDE | AERIAL-LF ADE/FDE |
| --- | ---: | ---: |
| All routes | 47.593 / 95.356 mm | 36.108 / 69.578 mm |
| Right | 62.128 / 138.098 mm | 42.308 / 84.457 mm |
| Straight | 27.946 / 36.856 mm | 21.853 / 33.487 mm |
| Left | 38.495 / 69.530 mm | 39.757 / 80.747 mm |

ADE is lower on 130/181 routes and FDE is lower on 113/181 routes. The
three individual AERIAL-LF OOF runs have mean and sample standard deviation
39.891 +/- 0.311 mm ADE and 72.778 +/- 1.259 mm FDE.

## Input Hashes

```text
features  324A480719D98ACA203E9B442B5D79A350B61BDA3977725D08FCE890B41803D9
split     505364B14FE7BCD993ACC997468C1D91B7723794F21E086E80451115094D4EBE
semantics 86343AF330CEAD33C8986B6A69E28B998A0CB6B338675017937FB3C8D570B82A
```

## Version Boundary

The archived training feature file contains explicit canonical DINO
replacements for 37 early scenes. A later export agrees with the archived
tokens for 77 scenes within absolute tolerance `1e-5`; scenes 0041--0060 differ
and the later tokens were not used for the reported F15 training. This release
therefore reproduces the archived experiment rather than silently substituting
newer feature assets.

The current aggregate comparison is an offline waypoint-prediction result. A
training-budget-matched vehicle-only continuation and a verified population
analysis of blocked/unblocked response remain separate experiments.

