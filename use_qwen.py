import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ARmodel import Transformer, ModelConfig # FIX: Import from ARmodel
import gc

def load_qwen_into_custom_model(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"Loading HF Model: {model_id}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu", # Load to CPU first to avoid VRAM fragmentation
        trust_remote_code=True
    )
    hf_config = hf_model.config
    
    print("Mapping Configuration...")
    config = ModelConfig(
        vocab_size=hf_config.vocab_size,
        num_dims=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=hf_config.num_key_value_heads,
        num_layers=hf_config.num_hidden_layers,
        ffn_hidden_dims=hf_config.intermediate_size,
        context_len=hf_config.max_position_embeddings,
        use_cache=True,
        use_flash=True, 
        attention_bias=True,       # Qwen uses bias for Q, K, V
        attention_out_bias=False,  # No bias for Output
        mlp_bias=False,            # No bias for MLP
        tie_weights=False,
        rmsnorm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
    )
    
    print("Initializing Custom Model...")
    custom_model = Transformer(config).to(torch.float16)
    
    hf_sd = hf_model.state_dict()
    custom_sd = custom_model.state_dict()
    
    mapping = {}
    
    # 1. Embeddings and Heads
    mapping["tokens_embedding.weight"] = "model.embed_tokens.weight"
    mapping["ll_head.weight"] = "lm_head.weight"
    mapping["norm.weight"] = "model.norm.weight"
    
    # 2. Layers
    for i in range(config.num_layers):
        # Attention Weights
        mapping[f"blocks.{i}.attention.wq.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        mapping[f"blocks.{i}.attention.wk.weight"] = f"model.layers.{i}.self_attn.k_proj.weight"
        mapping[f"blocks.{i}.attention.wv.weight"] = f"model.layers.{i}.self_attn.v_proj.weight"
        mapping[f"blocks.{i}.attention.wo.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"
        
        # Attention Biases
        if config.attention_bias:
            mapping[f"blocks.{i}.attention.wq.bias"] = f"model.layers.{i}.self_attn.q_proj.bias"
            mapping[f"blocks.{i}.attention.wk.bias"] = f"model.layers.{i}.self_attn.k_proj.bias"
            mapping[f"blocks.{i}.attention.wv.bias"] = f"model.layers.{i}.self_attn.v_proj.bias"
        
        if config.attention_out_bias:
            mapping[f"blocks.{i}.attention.wo.bias"] = f"model.layers.{i}.self_attn.o_proj.bias"

        # FFN Weights
        mapping[f"blocks.{i}.ffn.w1.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
        mapping[f"blocks.{i}.ffn.w3.weight"] = f"model.layers.{i}.mlp.up_proj.weight"
        mapping[f"blocks.{i}.ffn.w2.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
        
        if config.mlp_bias:
            mapping[f"blocks.{i}.ffn.w1.bias"] = f"model.layers.{i}.mlp.gate_proj.bias"
            mapping[f"blocks.{i}.ffn.w3.bias"] = f"model.layers.{i}.mlp.up_proj.bias"
            mapping[f"blocks.{i}.ffn.w2.bias"] = f"model.layers.{i}.mlp.down_proj.bias"
        
        # Norms
        mapping[f"blocks.{i}.norm_attention.weight"] = f"model.layers.{i}.input_layernorm.weight"
        mapping[f"blocks.{i}.norm_ffn.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"

    print("Copying Weights...")
    for custom_key, hf_key in mapping.items():
        if hf_key in hf_sd and custom_key in custom_sd:
            if custom_sd[custom_key].shape != hf_sd[hf_key].shape:
                print(f"Shape mismatch: {custom_key} {custom_sd[custom_key].shape} vs {hf_key} {hf_sd[hf_key].shape}")
                continue
            with torch.no_grad():
                custom_sd[custom_key].copy_(hf_sd[hf_key])
        else:
            if hf_key not in hf_sd:
                print(f"WARNING: HF Key {hf_key} not found")
            if custom_key not in custom_sd:
                print(f"WARNING: Custom Key {custom_key} not found (Check Bias configs)")

    print("Weights Loaded Successfully.")
    
    del hf_model
    del hf_sd
    gc.collect()
    torch.cuda.empty_cache()
    
    return custom_model.to("cuda")

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    with torch.no_grad():
        for next_token in model.generate(input_ids, max_new_tokens):
            word = tokenizer.decode(next_token[0])
            print(word, end="", flush=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
    print("\n\nDone.")

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load_qwen_into_custom_model(MODEL_ID)
    
    prompt = "Tell me a joke about Python programming."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    generate_text(model, tokenizer, text, max_new_tokens=100)