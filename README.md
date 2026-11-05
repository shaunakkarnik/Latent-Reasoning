# Geometry-supervised latent reasoning

Investigating whether geometric supervision improves spatial reasoning in vision-language models.

The method trains Qwen2.5-VL to reason through continuous latent vectors supervised by a frozen VGGT model. VGGT features from the same input image are spatially pooled and projected into the student's embedding space.

Training has three stages:

1. **Oracle warmup:** learn to answer using projected VGGT vectors.
2. **Geometry alignment:** learn to predict those vectors while retaining answer supervision.
3. **Answer refinement:** generate recurrent latents and optimize them through answer loss alone.

At inference, the student generates and consumes its own latent vectors before answering, without running VGGT.

See [USAGE.md](USAGE.md) for setup, training, and evaluation instructions.
See [SPATIALLADDER.md](SPATIALLADDER.md) for server-side dataset preparation and
the SpatialLadder / SPBench-SI comparison workflow.
