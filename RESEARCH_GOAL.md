# Research goal: Geometry-supervised continuous latent reasoning for VLMs

Investigate whether supervising a vision-language model's continuous reasoning latents with **VGGT-derived geometric features** improves spatial reasoning, particularly relative spatial relationships and perspective-taking.

The method follows an LVR/LanteRn-style framework: the student generates continuous vectors and feeds them back as subsequent input embeddings, bypassing discrete vocabulary decoding. The proposed change replaces crop-derived visual supervision with geometric features from a pretrained VGGT teacher.

The central hypothesis is:

> Geometric oracle supervision can make continuous reasoning latents more useful for spatial tasks, and recurrent latent feedback can help the model use that geometry in subsequent computation.

VGGT receives the same source images as the student. It supplies a geometrically informed representation and learned priors, rather than additional scene observations. Single-view features cannot establish unseen scene details.

## Method

Use **Qwen2.5-VL-3B-Instruct**, training the language model, input embeddings, and output weights. Keep its vision encoder and existing visual merger frozen. VGGT also remains frozen.

The initial oracle pipeline is:

`whole image -> VGGT aggregator patch features -> 4x4 spatial pooling -> normalization -> linear adapter -> 16 oracle vectors`

Remove camera/register tokens before pooling. Preserve the 2D grid in row-major order and normalize channels using training-only statistics. Cache pooled teacher features, not trainable projected targets. Begin with the final aggregator layer and one image per example.

Oracle vectors describe the whole observed scene and are independent of the question. Use one fixed-length latent block before the answer. Its spatial slots are representation targets; they are not assumed to be semantic reasoning steps.

## Training and inference

1. **Oracle warmup:** supply projected VGGT vectors as latent inputs and train the adapter and student with answer loss. This establishes the VGGT-to-student interface; it is a method-specific choice, not LVR's first stage.
2. **Geometry alignment:** freeze the adapter and train the student and latent prediction head with answer loss plus latent MSE. Predict each target before consuming it: the start-token state predicts the first vector, and the state after each oracle input predicts the next.
3. **Answer refinement:** replace oracle inputs with the student's own recurrently generated vectors. Remove alignment loss and backpropagate answer loss through every latent step without detaching feedback. Keep the oracle adapter frozen.

In alignment, use `L = L_answer + lambda * mean(MSE(predicted_latents, oracle_latents))`. Mask prompt and latent slots from vocabulary cross-entropy; supervise the end boundary and answer tokens. Average alignment loss over slots and channels and tune its weight on validation scenes.

At inference, generate and consume exactly 16 latent vectors, append the end boundary, and decode the answer. VGGT and its feature cache are absent from this path. Refinement allows the latents to adapt to answering, so retained geometric content must be measured rather than assumed.

## Data and evaluation

Select a spatial QA corpus with reliable scene identifiers. Initial candidates are a single-image subset of [SpatialLadder-26k](https://huggingface.co/datasets/hongxingli/SpatialLadder-26k) or [SAT-v2](https://huggingface.co/datasets/array/SAT-v2), with [SAT perspective data](https://huggingface.co/datasets/array/SAT_perspective) as a perspective-taking candidate.

Split by scene, grouping all views and questions together. Exclude helper views and answer-bearing annotations from teacher inputs. Any later multi-image extension must use the same legitimate views for teacher and student and combine only views of the same scene. Reserve [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) and [BLINK spatial subsets](https://zeyofu.github.io/blink/) for external evaluation.

The research should establish three separate conclusions: whether geometry supervision helps, whether generated latents retain oracle benefits, and whether recurrent feedback contributes to answering.

Use matched prompts, pixels, splits, answer budgets, trainable modules, and tuning budgets for these comparisons:

- Answer-only fine-tuning versus geometry-supervised latent reasoning.
- Crop-supervised versus VGGT-supervised recurrent latents.
- Oracle versus generated latents, including before and after refinement.
- Generated latents versus fixed dummy vectors, norm-matched noise, and wrong-scene vectors.
- Geometry alignment as an auxiliary loss without latent feedback.
- Matched recurrent computation with alignment supervision disabled.

Combine interventions with geometric probes and controlled scene changes. A drop after replacing latents can reflect distribution shift; latent sensitivity alone does not establish causal use of geometry. Low alignment error also does not establish usefulness. If oracle conditioning fails to help, examine the teacher representation, compression, grounding, and task difficulty before scaling.

Report per-task accuracy, scene-level paired confidence intervals, multiple seeds for the main comparison, oracle/generated gaps, latent norms and variance, per-slot prediction errors, and training/inference compute.

## Research sequence

1. Select data and scene splits; establish untouched-student inference and profile the chosen configuration.
2. Run a tiny-set learning check, then a small complete warmup/alignment/refinement run. Verify intended gradients, checkpoint resumption, and teacher-free generation without requiring an accuracy improvement.
3. Measure oracle usefulness and generated-latent degradation; assess how answer refinement changes both accuracy and geometric content.
4. Run matched baselines, causal interventions, multiple seeds, and external evaluation. Explore latent counts or compression changes only after the initial configuration is understood.

Keep the prototype standalone. Defer RL, dynamic latent stopping, additional VLM families, and large architecture searches until the supervised experiments answer the core questions.

References: [LVR](https://arxiv.org/abs/2509.24251), [LanteRn](https://github.com/GuilhermeViveiros/LantErn), [VGGT](https://github.com/facebookresearch/vggt), and [Mirage](https://vlm-mirage.github.io/).
