import argparse
import json
from pathlib import Path

import torch

from .backbone import load_student
from .baseline import AnswerOnlyReasoner
from .checkpoint import restore_checkpoint
from .config import LatentConfig, Stage
from .data import ExampleEncoder, read_manifest
from .evaluation import evaluate
from .geometry import VGGTExtractor
from .model import LatentReasoner
from .training import train


def load_run(checkpoint=None, config_path=None, *, device="cpu", load_weights=True, answer_only=False,
             dtype=torch.float32):
    if checkpoint and config_path:
        raise ValueError("A checkpoint supplies its configuration; do not also pass --config")
    if checkpoint:
        config = LatentConfig.load(Path(checkpoint) / "latent_config.json")
        metadata = json.loads((Path(checkpoint) / "metadata.json").read_text())
        checkpoint_answer_only = metadata.get("model_kind", "latent") == "answer_only"
        if answer_only and not checkpoint_answer_only:
            raise ValueError("Expected an answer-only checkpoint")
        answer_only = checkpoint_answer_only
    elif config_path:
        config = LatentConfig.load(config_path)
    else:
        config = LatentConfig()
    processor_path = Path(checkpoint) / "processor" if checkpoint else None
    options = {"add_latent_tokens": False} if answer_only else {}
    backbone, processor, start_id, end_id = load_student(
        config, dtype=dtype, device=device, processor_path=processor_path, **options)
    model = (AnswerOnlyReasoner(backbone, config) if answer_only
             else LatentReasoner(backbone, config, start_id, end_id))
    if checkpoint:
        metadata = json.loads((Path(checkpoint) / "metadata.json").read_text())
        model.set_stage(metadata["stage"])
        if load_weights:
            restore_checkpoint(checkpoint, model)
    return model, processor, ExampleEncoder(processor, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Geometry-supervised continuous latent reasoning")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-spatialladder", help="Download single-image training QA and SPBench-SI")
    prepare.add_argument("--output-dir", default="data/spatialladder")
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--training-revision", default="main")
    prepare.add_argument("--benchmark-revision", default="main")
    extract = sub.add_parser("extract", help="Cache frozen VGGT features; write an enriched manifest")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output-manifest", required=True)
    extract.add_argument("--cache-dir", required=True)
    extract.add_argument("--config")
    extract.add_argument("--teacher", default="facebook/VGGT-1B")
    extract.add_argument("--teacher-revision", default="main")
    extract.add_argument("--layer", type=int, default=-1)
    extract.add_argument("--fp32", action="store_true")
    training = sub.add_parser("train", help="Run a latent training stage or the answer-only sft baseline")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config")
    training.add_argument("--stage", choices=[s.value for s in Stage], required=True)
    training.add_argument("--output-dir", required=True)
    source = training.add_mutually_exclusive_group()
    source.add_argument("--init-from", help="Initialize weights for a new stage/run; reset optimizer")
    source.add_argument("--resume", help="Resume weights, optimizer, RNG, and example cursor")
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--learning-rate", type=float, default=1e-5)
    training.add_argument("--accumulation-steps", type=int, default=1)
    training.add_argument("--checkpoint-every", type=int, default=100)
    training.add_argument("--compact-checkpoints", action="store_true",
                          help="Keep the latest full recovery checkpoint; finish with weights only")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--max-prompt-tokens", type=int, default=4096)
    training.add_argument("--max-answer-tokens", type=int, default=256)
    evaluation = sub.add_parser("evaluate", help="Write held-out predictions with generated or oracle latents")
    eval_source = evaluation.add_mutually_exclusive_group(required=True)
    eval_source.add_argument("--checkpoint")
    eval_source.add_argument("--original-model", action="store_true", help="Evaluate the original Qwen model without latents")
    evaluation.add_argument("--config", help="Model configuration for --original-model")
    evaluation.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="val")
    evaluation.add_argument("--oracle", action="store_true")
    evaluation.add_argument("--max-new-tokens", type=int, default=64)
    evaluation.add_argument("--no-cache", action="store_true")
    generation = sub.add_parser("generate", help="Answer one image/question; no VGGT dependency or feature cache")
    generation.add_argument("--checkpoint", required=True)
    generation.add_argument("--image", required=True)
    generation.add_argument("--question", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=64)
    generation.add_argument("--no-cache", action="store_true")
    for command in (extract, training, evaluation, generation):
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.command == "prepare-spatialladder":
        from .spatialladder import prepare_spatialladder
        print(json.dumps(prepare_spatialladder(args.output_dir, val_fraction=args.val_fraction,
            seed=args.seed, training_revision=args.training_revision,
            benchmark_revision=args.benchmark_revision), indent=2))
        return
    if args.command == "extract":
        records = read_manifest(args.manifest)
        target = Path(args.output_manifest)
        if target.exists():
            raise FileExistsError(target)
        config = LatentConfig.load(args.config) if args.config else LatentConfig()
        extractor = VGGTExtractor(checkpoint=args.teacher, revision=args.teacher_revision, layer=args.layer,
                                  pool_shape=(config.pool_height, config.pool_width), device=args.device,
                                  dtype=torch.float32 if args.fp32 else torch.bfloat16)
        feature_paths = {}
        for record in records:
            image_path = record["image"]
            if image_path not in feature_paths:
                feature_paths[image_path] = str(extractor.cache(image_path, args.cache_dir))
            record["teacher_feature"] = feature_paths[image_path]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} examples to {target}")
        return
    if args.command == "train":
        checkpoint = args.resume or args.init_from
        if args.stage not in (Stage.WARMUP.value, Stage.SFT.value) and not checkpoint:
            raise ValueError("Alignment/refinement require --init-from or --resume")
        # Seed before initializing newly added boundary embeddings/heads.
        torch.manual_seed(args.seed)
        # Resume restores weights with optimizer/RNG in train(), in a single read.
        # New stages initialize weights here and start with a fresh optimizer.
        model, processor, encoder = load_run(
            checkpoint, args.config, device=args.device, load_weights=not bool(args.resume),
            answer_only=args.stage == Stage.SFT.value)
        if (args.stage == Stage.SFT.value) != isinstance(model, AnswerOnlyReasoner):
            raise ValueError("Cannot switch between answer-only and latent checkpoints")
        if args.init_from:
            allowed = {Stage.SFT: {Stage.SFT}, Stage.WARMUP: {Stage.WARMUP}, Stage.ALIGNMENT: {Stage.WARMUP, Stage.ALIGNMENT},
                       Stage.REFINEMENT: {Stage.ALIGNMENT, Stage.REFINEMENT}}
            if model.stage not in allowed[Stage(args.stage)]:
                raise ValueError("Stage order is warmup -> alignment -> refinement")
        encoder.max_prompt_tokens = args.max_prompt_tokens
        encoder.max_answer_tokens = args.max_answer_tokens
        train(model, processor, encoder, read_manifest(args.manifest), output_dir=args.output_dir,
              stage=args.stage, steps=args.steps, learning_rate=args.learning_rate,
              accumulation_steps=args.accumulation_steps, seed=args.seed,
              checkpoint_every=args.checkpoint_every, resume=args.resume,
              gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16,
              compact_checkpoints=args.compact_checkpoints)
        return
    if args.command == "evaluate":
        if args.original_model and args.oracle:
            raise ValueError("--oracle requires a latent checkpoint")
        if args.config and not args.original_model:
            raise ValueError("--config is only for --original-model; checkpoints supply their configuration")
        dtype = torch.bfloat16 if args.bf16 and torch.device(args.device).type == "cuda" else torch.float32
        model, processor, encoder = load_run(args.checkpoint, args.config, device=args.device,
            answer_only=args.original_model, dtype=dtype)
    else:
        model, processor, encoder = load_run(args.checkpoint, device=args.device)
    if args.command == "evaluate":
        print(json.dumps(evaluate(model, processor, encoder, read_manifest(args.manifest),
            output_path=args.output, split=args.split, oracle=args.oracle,
            max_new_tokens=args.max_new_tokens, use_cache=not args.no_cache, bf16=args.bf16), indent=2))
    elif args.command == "generate":
        inputs = encoder.prompt(args.image, args.question)
        result = model.generate(inputs, max_new_tokens=args.max_new_tokens,
            eos_token_ids=[processor.tokenizer.convert_tokens_to_ids("<|im_end|>")], use_cache=not args.no_cache)
        print(json.dumps({"answer": processor.tokenizer.decode(result.token_ids, skip_special_tokens=True),
                          "latent_count": result.latents.shape[1]}))


if __name__ == "__main__":
    main()
