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
