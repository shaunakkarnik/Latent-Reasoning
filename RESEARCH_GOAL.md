# Research goal: Geometry-supervised continuous latent reasoning for VLMs

Investigate whether supervising a vision-language model’s continuous reasoning latents with **VGGT-derived geometric features** improves spatial reasoning, particularly questions involving relative spatial relationships and perspective-taking.

The proposed method follows an **LVR/LanteRn-style latent reasoning framework**. During latent reasoning, the model generates continuous vectors and feeds them back as subsequent input embeddings, bypassing discrete vocabulary decoding. The key change is to replace crop-derived visual embeddings as oracle supervision targets with geometric features extracted by a pretrained VGGT model.

VGGT features would be compressed into a manageable sequence of tokens and projected into the MLLM’s embedding dimension. During teacher-forced training, these projected oracle vectors would serve as latent inputs, while an alignment objective trains the model to predict them. At inference, the model would generate and recurrently consume its own latent vectors without requiring VGGT.

The motivation comes from evidence that existing crop-supervised latent reasoning models often bypass their latent pathway because the oracle representations offer little additional computational benefit. VGGT supervision may make geometry more accessible than ordinary visual features, giving the model a stronger incentive to use its latents. When VGGT receives the same images as the student, this supplies a geometrically informed representation and learned priors, rather than genuinely new scene observations.

The central hypothesis is:

> **Geometric oracle supervision can make continuous reasoning latents more useful for spatial tasks, and recurrent latent feedback can help the model use that geometry in subsequent computation.**

The research should establish:

- Whether VGGT supervision improves over crop-based latent supervision.
- Whether the model causally uses the geometric content of its latents.
- Whether generated latents retain the benefits available with oracle latents.
