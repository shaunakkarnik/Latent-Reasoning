# Implementation plan: geometry-supervised continuous latent reasoning

Draft — November 2, 2026. Based on RESEARCH_GOAL.md and the initial design discussion. This is an implementation proposal; architecture choices marked provisional need discussion or pilot evidence.

### Accepted training schedule and implementation status

The subsequent design discussion accepted three explicit stages: (1) oracle-conditioned
answer warmup to train the VGGT feature adapter and student, (2) teacher-forced
geometry alignment plus answer loss with the oracle adapter frozen, and (3)
answer-only refinement using the student's own recurrently generated latents, with
gradients through every latent step and no alignment loss. This replaces the
conditional mixed-feedback proposal in section 5 as the initial training schedule.
Adapter warmup is a prototype-specific interface choice, not a claim about LVR's stages.

The standalone implementation now lives in `src/latent_geometry/`; see `README.md`
for commands, checkpoint behavior, and supported scope. Local tiny-Qwen checks
exercise all stages, shifted supervision, frozen parameters, 16-step teacher-free
generation, cached/reference equivalence, and exact optimizer resumption. They do
not establish the first research training milestone on pretrained models or a
selected spatial dataset. Initial training is single-image, microbatch one with
gradient accumulation, and single-process. GPU profiling and any required
distributed sharding remain cluster work.

## Confirmed constraints

- Cluster access to H100/H200 GPUs; simultaneous allocation, memory per GPU, storage, and job duration remain unknown.
- VGGT and the student receive the same source images and views. Each uses its own required preprocessing; record transformations and avoid exposing additional image regions to the teacher.
- Begin with pretrained VGGT intermediate features, rather than explicit depth/point/camera targets.
- No spatial dataset has been selected yet.
- Selected student: Qwen/Qwen2.5-VL-3B-Instruct with full language-model tuning, including language embeddings/output weights. Keep the student's vision encoder and existing visual merger frozen; VGGT also remains frozen.
- Accepted initial oracle pipeline: whole image -> frozen VGGT aggregator features -> remove camera/register tokens -> 4x4 spatial average pooling -> feature-space adapter -> 16 oracle latent vectors. Feature layer and adapter details use the configurable defaults below.
- Build a small standalone prototype; consult LVR/LanteRn code and selectively reuse useful ideas or components rather than importing their full training frameworks.
- The final student must answer using its own recurrently generated latents without running VGGT.
- Oracle latents represent the whole observed scene and are independent of the question; they are not per-step or question-selected targets.
- Use one continuous latent block before the answer for the first version.
- The first milestone is successful end-to-end training and usable checkpoint/inference plumbing. No accuracy gain or baseline improvement is required for this milestone.

## Recommended starting point

Build a standalone prototype around the pretrained student's supported model API, with separate VGGT extraction, oracle projection, latent prediction, and recurrent decoding modules. Inspect LVR/LanteRn for latent shifting, loss masking, embedding feedback, and multimodal cache/position handling. Selectively adapt useful code with source attribution and compatible licensing; avoid vendoring entire training frameworks. The README-level review is not yet a code audit.

| Approach | What we would build | Benefit | Cost |
| --- | --- | --- | --- |
| Standalone prototype | Latent input insertion, shifted latent loss, recurrent decoding, cache/position handling, and training integration | Small, understandable system with control over every component | Must establish that the latent machinery is correct and recreate a fair crop baseline |
| Adapt existing implementation | Reuse latent machinery; replace target extraction and add controlled experiments | Closer comparison with crop supervision and less initial infrastructure work | Inherited model patches, dependencies, and assumptions require inspection |

LanteRn documents a Qwen2.5-VL interface, continuous latent blocks, and crop-oriented training data. The official LVR implementation is also based on Qwen2.5-VL. These make Qwen2.5-VL a practical starting family, rather than a claim that it is the best available VLM. [LanteRn](https://github.com/GuilhermeViveiros/LantErn), [LVR](https://github.com/VincentLeebang/lvr).

Selected student: Qwen/Qwen2.5-VL-3B-Instruct with full language-model tuning. Freeze the student vision encoder and existing visual merger, and train all language-model weights plus the new latent prediction head. Train the VGGT-to-student feature adapter during oracle-conditioning warmup, then freeze it for latent-prediction training as described below. LoRA is not part of the initial configuration. Verify continuous-input support in the chosen model API and profile one full training step before fixing allocation or increasing model size. Keep initialization and trainable modules identical across comparison arms.

Verified reference configurations: LVR's [stage-1 script](https://github.com/VincentLeebang/lvr/blob/main/scripts/finetune_lvr_stage1_7b.sh) uses Qwen2.5-VL-7B-Instruct with frozen vision tower/merger and language-model tuning. LanteRn provides [3B](https://github.com/GuilhermeViveiros/LantErn/blob/main/scripts/finetune_lantern_sft_3b.sh) and [7B](https://github.com/GuilhermeViveiros/LantErn/blob/main/scripts/finetune_lantern_sft_7b.sh) Qwen2.5-VL-Instruct scripts. Its current scripts request 16 GPUs for 3B and 4 GPUs for 7B; these are configured experiment allocations, not measured minimum requirements. LVR declares NUM_DEVICES=8, but the shown launch does not use it to constrain device count, so it does not establish an eight-GPU requirement.

For our prototype, provisional planning allocations with BF16, frozen vision, checkpointing, small microbatches, bounded image/text lengths, and offline teacher extraction are:

| Student | LoRA tuning | Full language-model tuning with sharding where needed |
| --- | --- | --- |
| 3B | Start with 1 H100 80GB or 1 H200 141GB | Budget 2 H100 80GB or 1 H200 141GB |
| 7B | Start with 1 H100 80GB or 1 H200 141GB | Budget 4 H100 80GB or 2 H200 141GB |

These are engineering estimates, not benchmarked minima or runtime guarantees. Profile our actual code before reserving a long run. Memory depends on trainable parameter count, optimizer, image tokens, sequence length, microbatch, and whether feedback training retains recurrent graphs. Extract VGGT features in a separate job; it need not occupy another GPU during student training.

For the selected 3B full-tuning configuration, initially plan around one H200 141GB or two H100 80GB GPUs, subject to profiling. Use BF16, gradient checkpointing, and gradient accumulation; add optimizer/parameter sharding for the multi-GPU configuration as needed.

## 1. Establish data and baseline evaluation

Use a manageable SAT training subset as the first candidate, covering static relationships and perspective-taking. SAT provides spatial QA data and generation code, including perspective tasks. Inspect the actual samples and scene identifiers before committing. [SAT repository](https://github.com/arijitray1993/SAT).

- First use 32–128 examples for a tiny-set learning check, then roughly 1k questions for an end-to-end training run. Expand toward 5–10k only after the pipeline works; these are engineering sizes, not a power calculation.
- Split by scene, grouping every view and question from a scene together. Hold out validation scenes for model selection and test scenes for final reporting.
- Keep rendered alternate-view reasoning images out of VGGT inputs unless those same views are also legitimate student inputs. Answer-bearing traces and annotations must not enter target extraction.
- Reserve BLINK relative-depth, spatial-relation, and multi-view subsets for external evaluation. BLINK contains those task categories; it is an evaluation source here, not the training corpus. [BLINK](https://zeyofu.github.io/blink/).
- Process jointly in VGGT only images that depict views of the same scene. Unrelated panels or candidate images must not be treated as a reconstructed scene.
- Run untouched-student inference as a plumbing check. Defer answer-only fine-tuning and comparative performance experiments until after the first training milestone.

Deliverable: versioned dataset manifests, documented image/view policy, evaluation script, and baseline results.

Dataset shortlist: [SAT-v2](https://huggingface.co/datasets/array/SAT-v2) for training, beginning with static relative-depth and object-relation questions; [SAT perspective data](https://huggingface.co/datasets/array/SAT_perspective) as a later perspective-taking candidate, excluding helper views from teacher inputs under the same-images constraint; [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) for external spatial/depth/distance evaluation; and BLINK spatial subsets for a second external check. Keep CV-Bench and BLINK out of training.

Additional training candidates remain under consideration: [SpatialLadder-26k](https://huggingface.co/datasets/hongxingli/SpatialLadder-26k) (single-image/multi-view/video QA, used by SpatialLadder-3B); [SPAR-7M](https://github.com/LogosIntelligenceLab/SPAR) (broad spatial perception/reasoning); [MindCube training split](https://github.com/mll-lab-nu/MindCube) (limited-view spatial mental modeling); [CA-VQA](https://machinelearning.apple.com/research/mm-spatial) (MM-Spatial's indoor spatial training data); and [SpaceR-151k](https://github.com/OuyangKun10/SpaceR) (video spatial training). These are a shortlist, not an approved mixture. For minimal initial complexity, compare a single-image SpatialLadder QA subset with SAT-v2 before choosing; keep video/multi-view support for a subsequent step if not needed by the selected subset.

Task choice initially changes which information compression must retain, not the VGGT checkpoint or a proven task-specific feature layer. Relative depth needs local spatial detail; object relationships need object identity and location to remain associated; perspective-taking needs spatial layout and reference-frame information. Preserve a spatial grid rather than global averaging. Intermediate features are not guaranteed to preserve every ability, and single-view features cannot establish unseen scene details. Start with a generic whole-scene grid and defer task-specific compression searches.

## 2. Extract and compress geometric oracle features

Freeze VGGT and run it offline. Its documented aggregator interface exposes intermediate tokens and the patch-token start index. Select patch tokens explicitly rather than accidentally including camera/register tokens. [VGGT usage](https://github.com/facebookresearch/vggt), [aggregator implementation](https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py).

Accepted initial design, with configurable implementation defaults:

The user accepted whole-scene VGGT features with spatial average pooling and a feature-space adapter; dataset selection remains open. Reference-code inspection distinguishes LVR's main ROI token selection from LanteRn's fixed-length pooling: [LVR](https://arxiv.org/html/2509.24251#S3.SS2) selects already encoded visual tokens corresponding to annotated regions, while [LanteRn's compression helper](https://github.com/GuilhermeViveiros/LantErn/blob/main/src/models/utils.py) encodes crop inputs, groups consecutive output tokens, and averages each group into one of K targets (dropping trailing tokens to make the count divisible). Our implementation will preserve the two-dimensional grid and avoid silently dropping patches. This is an adaptation of LanteRn's compression rather than an exact copy. A learned feature-space adapter remains separate from reducing token count, even if teacher and student channel dimensions happen to match.

1. Default to the final aggregator layer; confirm its shape at the pinned revision and keep layer selection configurable.
2. Spatially pool each image's patch grid to a fixed grid, retaining a deterministic row-major order and view order. Begin with 16 total tokens for a single image; define a fixed total budget before multi-image experiments.
3. Normalize feature channels using training-only statistics and project to the student's embedding dimension.
4. Use stable slot/view information and inspect target norms against the student's input embedding norms.

Default to a small linear feature-space adapter for the initial implementation; retain the oracle-conditioning warmup and subsequent freeze described below. Start with single-image examples. Keep feature extraction and data collation modular so same-scene multi-image support can be added after a dataset is chosen. Dataset-independent work can use a tiny image/question/answer fixture to validate shapes, losses, gradients, checkpointing, and decoding without presenting that fixture as research training data.

Cache pooled teacher features, not full dense features, after checking that pooling preserves useful information. Store image hashes, scene/view IDs, checkpoint and code revision, layer, dtype, preprocessing, pooling specification, and feature shape. Profile extraction throughput and storage on a small batch before scaling.

The adapter needs a stable training schedule. First train the projection and student interface using oracle-conditioned answer loss. Then freeze the projection before introducing latent prediction loss. Detaching targets alone does not prevent a trainable projector from collapsing across optimization steps. Monitor across-example target variance and geometry probes; frozen projected targets are stable but are not automatically geometrically informative.

If uniform pooling fails the oracle test, examine finer grids or a learned resampler before scaling training. Question-conditioned selection is a later design choice, not an assumed part of this first version.

Deliverable: deterministic extractor, versioned feature cache, compressor/projector, and feature-quality report.

## 3. Validate oracle conditioning; assess usefulness later

Train an oracle-conditioned answer model with the compressed VGGT vectors supplied in a latent block. For the first milestone, verify finite losses, expected gradients and parameter updates, and tiny-set learning. This stage initializes the projection before freezing it for latent prediction.

After the first milestone, evaluate correct oracle vectors against matched dummy and wrong-scene vectors, and train a matched crop-feature arm. Held-out answer improvement and sensitivity to oracle content are research questions, not requirements for getting the prototype trained. Low alignment error is not evidence of usefulness. The bypass and latent-collapse findings in [What's Holding Back Latent Visual Reasoning?](https://arxiv.org/abs/2605.18445) motivate these later diagnostics.

If later experiments show no reliable oracle benefit, inspect task difficulty, teacher quality on synthetic imagery, compression, object grounding, and representation accessibility before scaling the research runs. This does not block the initial end-to-end implementation.

Initial deliverable: working oracle conditioning and projection checkpoint. Later deliverable: oracle-versus-control results informing representation/data revisions.

## 4. Add teacher-forced latent prediction

For the first version, use one fixed-length latent block before the answer. This avoids requiring reasoning traces and makes token-budget comparisons straightforward. A sequence of spatial feature slots is initially a representation reconstruction target; it does not by itself establish a sequence of semantic reasoning steps.

Let x contain the student image tokens and question, t_1...t_K be frozen projected oracle vectors, and y be the answer. The input layout is:

`x, <latent_start>, t_1, ..., t_K, <latent_end>, y`

Predict each t_i from the hidden state immediately BEFORE its input position: the start-token state predicts t_1, and the state after consuming t_(i-1) predicts t_i. Never regress a hidden state to an oracle vector already visible at that same position.

Use a small prediction head g mapping the student hidden state into the same embedding space used by the oracle vectors and recurrent feedback:

`L = L_answer + lambda_latent * mean_i MSE(g(h_(i-1)), stop_gradient(t_i))`

- Mask latent slots from vocabulary cross-entropy. Handle boundary and answer tokens explicitly.
- Average latent loss over tokens and dimensions, and answer loss over supervised tokens; tune the relative weight on validation data.
- Begin with MSE; add a cosine term only if scale/direction diagnostics justify it.
- Keep VGGT and the oracle projection frozen during this stage. Do not cache trainable projected targets.
- Evaluate oracle-conditioned and generated-latent answers separately at every checkpoint.

Deliverable: supervised model with correctly shifted prediction targets and per-task evaluation.

## 5. Implement recurrent generation and address the training gap

At inference, consume x and the start token, predict z_1 = g(h_start), append z_1 as the next input embedding, and repeat for exactly K vectors. Append the end token and decode the answer normally. VGGT and its feature cache must be absent from this execution path.

Validate custom generation against a simple uncached reference. Inspect causal masks, multimodal positions, padding, cache length, dtype, and normalization. Confirm that generated vector i actually affects subsequent computation.

If oracle answers are strong but generated answers are weak, introduce a short mixed-feedback stage with increasing fractions of generated latent prefixes. Retain oracle alignment targets and answer supervision. Specify whether gradients pass through recurrent feedback; start with a small fully generated unroll to measure memory before choosing a training strategy. Match any additional training budget in the baselines.

Deliverable: teacher-free inference, measured oracle/generated gap, and a documented feedback-training decision.

## 6. Test the research claims (after the first milestone)

Run these comparisons in stages, using matched prompts, input pixels, splits, answer budgets, tuning budgets, and trainable modules. Record differences in training/inference compute rather than assuming they are equal.

| Comparison | What it establishes |
| --- | --- |
| Answer-only fine-tuning | Benefit beyond ordinary spatial-task training |
| Crop-supervised recurrent latents | Benefit of the proposed teacher representation |
| VGGT oracle versus generated latents | Representation utility versus prediction/rollout gap |
| Generated versus fixed dummy or matched-norm noise slots | Content contribution beyond extra token positions |
| Correct versus wrong-scene latents | Dependence on scene-specific information |
| Geometry prediction as auxiliary loss only, with no feedback | Whether gains can be explained by representation training alone |
| Matched recurrence arm with alignment loss disabled | Effect of geometry supervision beyond extra computation |

Use within-model interventions as well as separately trained baselines. A replacement-induced accuracy drop alone can reflect distribution shift. Combine norm-matched substitutions, scene swaps, geometric probes, and a small controlled set where a spatial relationship changes while irrelevant properties remain similar. Do not equate generic latent sensitivity with causal use of geometry.

Track answer accuracy by task and scene; paired confidence intervals with scene-level resampling; multiple seeds for the main comparison; per-slot rollout errors, latent norms and across-example variance; oracle/generated gap; peak memory, training throughput, teacher preprocessing cost, and inference latency. Vary K over a small range only after the first configuration works.

Deliverable: a reproducible result table supporting three separate conclusions: geometry usefulness, generated-latent quality, and contribution of recurrent feedback.

## Implementation milestones and scope

1. **Data/model audit:** select and pin the student checkpoint and model-library version, inspect relevant LVR/LanteRn reference code, select dataset and splits, profile a batch, and reproduce baseline inference.
2. **Oracle pilot:** build extraction/cache/compression; verify oracle conditioning and train the projection.
3. **Latent student:** implement shifted loss and cached recurrent decoding; verify with an uncached reference and a tiny-set overfit check.
4. **First milestone complete:** finish a small training run with finite answer/alignment losses, demonstrate tiny-set learning, confirm intended modules update and frozen modules remain frozen, save/reload/resume a checkpoint, and generate an answer through exactly K recurrent latent steps without VGGT. Log held-out outputs for inspection without requiring a performance threshold.
5. **Later feedback pilot:** measure rollout degradation; add mixed-feedback training if warranted.
6. **Later main experiments:** run matched baselines, interventions, seeds, and external evaluation.

Proposed modules: data manifest/collator, VGGT extractor/cache, oracle compressor/projector, student latent interface, training stages, recurrent decoder, and evaluation/intervention runner. Keep the standalone student interface small so crop and geometry targets use the same training and inference code. Add distributed-training infrastructure only as allocation and profiling require it.

Do not begin with GRPO, dynamic latent stopping, many VLM families, or a large compression architecture search. Add those only after the supervised prototype answers the core research questions. Calendar estimates depend on allocation and measured throughput.

## Questions to resolve next

1. How many H100/H200 GPUs can you reserve simultaneously, with what memory, job limits, and shared storage?
2. Which first training corpus should we use: a single-image spatial QA subset of SpatialLadder-26k, SAT-v2, or another candidate above? Keep CV-Bench/BLINK reserved for later evaluation.
3. Before cluster execution: what scheduler, environment setup, checkpoint/data paths, and launch procedure should be used? These do not block local implementation of model components.
