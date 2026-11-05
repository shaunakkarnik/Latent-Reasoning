import argparse
import json
from pathlib import Path

import torch

from .backbone import load_student
from .checkpoint import restore_checkpoint
from .config import LatentConfig, Stage
from .data import ExampleEncoder, read_manifest
from .evaluation import evaluate
from .geometry import VGGTExtractor
from .model import LatentReasoner
from .training import train


def load_run(checkpoint=None, config_path=None, *, device="cpu", load_weights=True):
    if checkpoint and config_path:
        raise ValueError("A checkpoint supplies its configuration; do not also pass --config")
    if checkpoint:
        config = LatentConfig.load(Path(checkpoint) / "latent_config.json")
    elif config_path:
        config = LatentConfig.load(config_path)
    else:
        config = LatentConfig()
    processor_path = Path(checkpoint) / "processor" if checkpoint else None
    backbone, processor, start_id, end_id = load_student(
        config, dtype=torch.float32, device=device, processor_path=processor_path)
    model = LatentReasoner(backbone, config, start_id, end_id)
    if checkpoint:
        metadata = json.loads((Path(checkpoint) / "metadata.json").read_text())
        model.set_stage(metadata["stage"])
        if load_weights:
            restore_checkpoint(checkpoint, model)
    return model, processor, ExampleEncoder(processor, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Geometry-supervised continuous latent reasoning")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Cache frozen VGGT features; write an enriched manifest")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output-manifest", required=True)
    extract.add_argument("--cache-dir", required=True)
    extract.add_argument("--config")
    extract.add_argument("--teacher", default="facebook/VGGT-1B")
    extract.add_argument("--teacher-revision", default="main")
    extract.add_argument("--layer", type=int, default=-1)
    extract.add_argument("--fp32", action="store_true")
    training = sub.add_parser("train", help="Run one of the three training stages")
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
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--max-prompt-tokens", type=int, default=4096)
    training.add_argument("--max-answer-tokens", type=int, default=256)
    evaluation = sub.add_parser("evaluate", help="Write held-out predictions with generated or oracle latents")
    evaluation.add_argument("--checkpoint", required=True)
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
        if args.stage != Stage.WARMUP.value and not checkpoint:
            raise ValueError("Alignment/refinement require --init-from or --resume")
        # Seed before initializing newly added boundary embeddings/heads.
        torch.manual_seed(args.seed)
        # Resume restores weights with optimizer/RNG in train(), in a single read.
        # New stages initialize weights here and start with a fresh optimizer.
        model, processor, encoder = load_run(
            checkpoint, args.config, device=args.device, load_weights=not bool(args.resume))
        if args.init_from:
            allowed = {Stage.WARMUP: {Stage.WARMUP}, Stage.ALIGNMENT: {Stage.WARMUP, Stage.ALIGNMENT},
                       Stage.REFINEMENT: {Stage.ALIGNMENT, Stage.REFINEMENT}}
            if model.stage not in allowed[Stage(args.stage)]:
                raise ValueError("Stage order is warmup -> alignment -> refinement")
        encoder.max_prompt_tokens = args.max_prompt_tokens
        encoder.max_answer_tokens = args.max_answer_tokens
        train(model, processor, encoder, read_manifest(args.manifest), output_dir=args.output_dir,
              stage=args.stage, steps=args.steps, learning_rate=args.learning_rate,
              accumulation_steps=args.accumulation_steps, seed=args.seed,
              checkpoint_every=args.checkpoint_every, resume=args.resume,
              gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16)
        return
    model, processor, encoder = load_run(args.checkpoint, device=args.device)
    if args.command == "evaluate":
        print(json.dumps(evaluate(model, processor, encoder, read_manifest(args.manifest),
            output_path=args.output, split=args.split, oracle=args.oracle,
            max_new_tokens=args.max_new_tokens, use_cache=not args.no_cache), indent=2))
    elif args.command == "generate":
        inputs = encoder.prompt(args.image, args.question)
        result = model.generate(inputs, max_new_tokens=args.max_new_tokens,
            eos_token_ids=[processor.tokenizer.convert_tokens_to_ids("<|im_end|>")], use_cache=not args.no_cache)
        print(json.dumps({"answer": processor.tokenizer.decode(result.token_ids, skip_special_tokens=True),
                          "latent_count": result.latents.shape[1]}))


if __name__ == "__main__":
    main()
