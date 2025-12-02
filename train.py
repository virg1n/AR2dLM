import os
import gc
import argparse
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ARmodel import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader


def load_qwen_into_custom_model(
    model_id: str = "Qwen/Qwen2.5-Coder-0.5B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mask_token_id: int = 0,
) -> Tuple[Transformer, AutoTokenizer, int, int]:
    """
    Load a HuggingFace Qwen model, ensure a dedicated <MASK> token exists,
    build the matching custom Transformer, and copy over the weights.

    Returns:
        custom_model, tokenizer, mask_token_id, vocab_size_total
    """
    print(f"Loading HF model: {model_id}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    hf_config = hf_model.config

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.mask_token is None:
        vocab = tokenizer.get_vocab()
        if "<MASK>" in vocab:
            print("Found existing '<MASK>' token in vocab, using it as mask_token.")
            tokenizer.mask_token = "<MASK>"
        else:
            print("No mask_token found. Adding '<MASK>' as mask_token.")
            special_tokens = {"mask_token": "<MASK>"}
            num_added = tokenizer.add_special_tokens(special_tokens)
            print(f"Added {num_added} new special token(s): {special_tokens}")

        hf_model.resize_token_embeddings(len(tokenizer))
    else:
        print(f"Tokenizer already has a mask token: {tokenizer.mask_token}")

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("<MASK>")
    print("Using mask_token_id:", mask_token_id)

    vocab_size_total = hf_model.get_input_embeddings().weight.size(0)
    print("HF embedding vocab size (total):", vocab_size_total)
    print("tokenizer.vocab_size (base):", tokenizer.vocab_size)
    print("len(tokenizer) (total):", len(tokenizer))

    model_config = ModelConfig(
        vocab_size=vocab_size_total,
        num_dims=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        num_layers=hf_config.num_hidden_layers,
        ffn_hidden_dims=hf_config.intermediate_size,
        context_len=hf_config.max_position_embeddings,
        # architecture flags
        use_cache=False,                # disable cache during training
        use_flash=True,                 # enable Flash attention if available
        attention_bias=True,            # Qwen uses bias in q/k/v projections
        attention_out_bias=False,       # usually no bias on o_proj
        mlp_bias=False,
        tie_weights=False,
        rmsnorm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
        rope_theta=getattr(hf_config, "rope_theta", 1e6),
        mask_token_id=mask_token_id,
    )

    print(f"Initializing custom Transformer with config: {model_config}")
    custom_model = Transformer(model_config).to(dtype=torch.float16)

    hf_sd = hf_model.state_dict()
    custom_sd = custom_model.state_dict()

    mapping = {}

    # Embeddings + output head + final norm
    mapping["tokens_embedding.weight"] = "model.embed_tokens.weight"
    mapping["ll_head.weight"] = "lm_head.weight"
    mapping["norm.weight"] = "model.norm.weight"

    # Transformer blocks
    for i in range(model_config.num_layers):
        # Attention projections
        mapping[f"blocks.{i}.attention.wq.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        mapping[f"blocks.{i}.attention.wk.weight"] = f"model.layers.{i}.self_attn.k_proj.weight"
        mapping[f"blocks.{i}.attention.wv.weight"] = f"model.layers.{i}.self_attn.v_proj.weight"
        mapping[f"blocks.{i}.attention.wo.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"

        # Attention biases
        if model_config.attention_bias:
            mapping[f"blocks.{i}.attention.wq.bias"] = f"model.layers.{i}.self_attn.q_proj.bias"
            mapping[f"blocks.{i}.attention.wk.bias"] = f"model.layers.{i}.self_attn.k_proj.bias"
            mapping[f"blocks.{i}.attention.wv.bias"] = f"model.layers.{i}.self_attn.v_proj.bias"

        if model_config.attention_out_bias:
            mapping[f"blocks.{i}.attention.wo.bias"] = f"model.layers.{i}.self_attn.o_proj.bias"

        # MLP projections
        mapping[f"blocks.{i}.ffn.w1.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
        mapping[f"blocks.{i}.ffn.w3.weight"] = f"model.layers.{i}.mlp.up_proj.weight"
        mapping[f"blocks.{i}.ffn.w2.weight"] = f"model.layers.{i}.mlp.down_proj.weight"

        if model_config.mlp_bias:
            mapping[f"blocks.{i}.ffn.w1.bias"] = f"model.layers.{i}.mlp.gate_proj.bias"
            mapping[f"blocks.{i}.ffn.w3.bias"] = f"model.layers.{i}.mlp.up_proj.bias"
            mapping[f"blocks.{i}.ffn.w2.bias"] = f"model.layers.{i}.mlp.down_proj.bias"

        # Layer norms
        mapping[f"blocks.{i}.norm_attention.weight"] = f"model.layers.{i}.input_layernorm.weight"
        mapping[f"blocks.{i}.norm_ffn.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"

    print("Copying weights...")
    missing_in_hf = []
    missing_in_custom = []

    for custom_key, hf_key in mapping.items():
        if hf_key in hf_sd and custom_key in custom_sd:
            if custom_sd[custom_key].shape != hf_sd[hf_key].shape:
                print(
                    f"Shape mismatch for {custom_key}: "
                    f"{custom_sd[custom_key].shape} vs HF {hf_key} {hf_sd[hf_key].shape}"
                )
                continue
            with torch.no_grad():
                custom_sd[custom_key].copy_(hf_sd[hf_key])
        else:
            if hf_key not in hf_sd:
                missing_in_hf.append(hf_key)
            if custom_key not in custom_sd:
                missing_in_custom.append(custom_key)

    if missing_in_hf:
        print("WARNING: some HF keys not found:", missing_in_hf)
    if missing_in_custom:
        print("WARNING: some custom keys not found:", missing_in_custom)

    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    custom_model = custom_model.to(device=device, dtype=dtype)

    return custom_model, tokenizer, mask_token_id, vocab_size_total


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Language Model adapted from Qwen.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-0.5B")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./fine_code", 
        help="Path to tokenized Datatrove dataset (folder that contains .ds files).",
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Training compute dtype.",
    )
    parser.add_argument("--mask_token_id", type=int, default=0)
    parser.add_argument("--adap_factor", type=float, default=0.9)
    parser.add_argument("--num_epochs_total", type=int, help="Alias for --num_epochs", default=None)

    args = parser.parse_args()

    if args.num_epochs_total is not None:
        args.num_epochs = args.num_epochs_total

    if torch.cuda.is_available():
        if args.use_ddp and "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            device = f"cuda:{local_rank}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    dtype = getattr(torch, args.precision)

    print(f"Using device: {device}, dtype: {dtype}")

    model, tokenizer, mask_token_id, vocab_size_total = load_qwen_into_custom_model(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        mask_token_id=args.mask_token_id,
    )
    print("model mask token is: ", mask_token_id)
    print("tokenier's mask token: ")
    print(tokenizer.convert_ids_to_tokens(151665))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    trainer_config = TrainerConfig(
        vocab_size=vocab_size_total,  # use total vocab size (matches embeddings & mask id)
        num_epochs=args.num_epochs,
        use_ddp=args.use_ddp,
        clean_cuda_cache=True,
        use_compile=True,
        use_dtype=args.precision,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        accumulation_steps=10,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        tokenized_dataset_path=args.dataset_path,
        path_to_checkpoints=args.output_dir,
        masked_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        adap_factor=args.adap_factor,
    )

    os.makedirs(trainer_config.path_to_checkpoints, exist_ok=True)
    log_dir = os.path.dirname(trainer_config.eval_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    trainer = Trainer(trainer_config, model, tokenizer)

    data_loader = DataLoader(
        trainer_config,
        rank=trainer.ddp_rank,
        world_size=trainer.ddp_world_size,
    )

    try:
        trainer.train(data_loader)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
