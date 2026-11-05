# Geometry-supervised continuous latent reasoning for VLMs

Investigate whether **VGGT-derived geometric supervision** improves spatial reasoning in VLMs, particularly relative relationships and perspective-taking.

Following LVR/LanteRn, the student generates continuous vectors and feeds them back as input embeddings before answering. The proposed method replaces crop-derived supervision with geometric features from frozen VGGT, using the same source image as the student.

**Hypothesis:** geometric supervision makes recurrent latents more useful for spatial reasoning.

## Method

Use Qwen2.5-VL-3B-Instruct with a trainable language model and frozen vision encoder/merger. Pool VGGT patch features to a 4×4 grid, normalize, and linearly project them into 16 question-independent oracle vectors.

## Training and inference

1. **Warmup:** train the adapter and student to answer using oracle vectors.
2. **Alignment:** freeze the adapter; train the student and latent head with answer loss plus MSE against each oracle vector, predicted before it is consumed.
3. **Refinement:** use generated recurrent latents and answer loss alone, with gradients through all latent steps.

At inference, the student generates and consumes 16 latent vectors, then answers without VGGT.
The fixed latent-end boundary is supplied as input, not supervised as a prediction.
Answer loss covers the answer tokens and the generated assistant terminator.

## Data and evaluation

Train on the single-image spatial reasoning subset of SpatialLadder-26k. Hold out
training rooms for validation, grouping all ScanNet rescans of each room together.
Use SPBench-SI as the held-out, in-domain test benchmark, with multiple-choice
accuracy and numerical mean relative accuracy, reported by task and as a task
macro average.

Compare against the original Qwen2.5-VL-3B-Instruct model and an answer-only
fine-tuned version trained on the same training split without latent vectors.
Keep the vision encoder/merger frozen for both fine-tuned methods. Use identical
direct-answer question prompts and image preprocessing across comparisons.
Match distinct training questions and total example presentations: each latent
stage makes one pass over the chosen training set, and SFT makes three passes.
Pilot runs use one shared, seeded subset; full runs use the entire training split.
