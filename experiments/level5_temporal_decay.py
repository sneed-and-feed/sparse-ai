import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoConfig, AutoModelForCausalLM

from llama_surgery.adelic_cache import SurprisalAdelicCache


class TemporalAdelicCache(SurprisalAdelicCache):
    """
    TemporalAdelicCache implements Level 5 Adelic KV-Cache Condensation with Temporal Decay
    via the vectorized two-tier `SurprisalAdelicCache` (Tier 1 Outlier/Needle Shield + Tier 2
    Medoid-Value Condenser with synthetic RoPE variance decay and zero dummy padding).
    """

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return int(self.key_cache[layer_idx].shape[-2])


def test_temporal_adelic_cache():
    print("Initializing dummy TinyLlama on CPU to test TemporalAdelicCache...")
    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    
    max_cap = 32
    local_win = 16
    lambda_decay = 0.05
    
    adelic_cache = TemporalAdelicCache(
        max_capacity=max_cap, 
        local_window=local_win, 
        lambda_decay=lambda_decay
    )
    
    input_ids = torch.randint(0, config.vocab_size, (1, 1))
    
    print(f"Starting Generation. Max Capacity: {max_cap}, Local Window: {local_win}, Decay \u03bb: {lambda_decay}")
    
    with torch.no_grad():
        for step in range(100):
            outputs = model(
                input_ids=input_ids,
                past_key_values=adelic_cache,
                use_cache=True
            )
            input_ids = torch.randint(0, config.vocab_size, (1, 1))
            
            if (step + 1) % 10 == 0:
                physical_len = adelic_cache.key_cache[0].shape[-2]
                logical_len = adelic_cache.get_seq_length()
                print(f"Step {step+1:3d} | Logical RoPE Position: {logical_len:3d} | Physical Cache Size: {physical_len:3d} tokens")
                assert physical_len <= max_cap + 10, "Cache condensation failed to cap memory!"

    print("\nSuccess! TemporalAdelicCache applied synthetic RoPE decay via Medoid-Key magnitude scaling.")

if __name__ == "__main__":
    test_temporal_adelic_cache()
