# Local implementation validation

Validated November 2, 2026, on CPU with Python 3.11.5, PyTorch 2.6.0, and
Transformers 4.57.1. Dependencies were installed in an isolated environment at
`/private/tmp/latent-reasoning-venv`; no pretrained model weights were downloaded.

## Automated checks

`python -m pytest -q`: **21 passed**, including the optional VGGT integration test.
Without the `teacher` extra, that one integration test is skipped.

The tests check:

- Qwen image embeddings and multimodal positions against its native forward.
- Correctly shifted and causal latent alignment targets and vocabulary masking.
- Expected gradients and updates in all three stages; frozen parameters unchanged.
- Answer gradients reaching every generated latent in refinement, including with
  non-reentrant gradient checkpointing enabled.
- Cached/uncached agreement for text and image prompts, repeated calls, and oracle conditioning.
- Exactly 16 generated latents with no oracle access.
- Checkpoint reload, optimizer/RNG restoration, and identical CPU parameters after
  interrupted versus uninterrupted gradient-accumulated training.
- Spatial pooling, feature normalization, image/cache mismatches, scene/image
  leakage checks, and incompatible teacher extraction policies.
- Actual upstream VGGT preprocessing and a tiny random upstream aggregator,
  using pinned source `a288dd0f14786c93483e45524328726ab7b1b4ce` without pretrained weights.

The 16-latent smoke run used 12 optimizer steps per stage on one synthetic example:

| Stage | Initial loss | Final loss |
| --- | ---: | ---: |
| Warmup | 4.1182 | 1.0815 |
| Alignment | 0.9254 | 0.1098 |
| Refinement | 1.8928 | 0.0171 |

The frozen vision encoder stayed unchanged, cached generation matched the
uncached reference, and checkpoint reload/resumption passed. These values
demonstrate tiny-set learning only; the features and model weights are synthetic.

## Processor integration

Separately downloaded **only** the actual Qwen2.5-VL-3B-Instruct processor/tokenizer
files. Its image/question chat encoding, answer encoding, added latent boundary
tokens, and processor save/reload were checked against a tiny random Qwen model
using the real vocabulary and matching 14px image patch size. A full refinement
forward/backward produced finite loss and nonzero latent-head gradients.

The fixture produced 35 prompt tokens, a `[1, 4, 8]` image patch grid, and three
answer tokens including the assistant terminator. Tokenizer vocabulary survived
save/reload exactly. This manual check requires downloaded processor files; it
is not part of the offline automated test suite.

## Still unverified

Pretrained Qwen/VGGT outputs, GPU execution and memory/throughput, spatial-task
accuracy, useful geometry retention, dataset-specific evaluation, and multi-GPU
training. The initial runner explicitly supports one process and one unpadded
example per forward, with gradient accumulation. Dataset selection and cluster
setup are required before an actual research training run.
