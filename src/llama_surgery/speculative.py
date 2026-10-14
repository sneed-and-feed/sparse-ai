"""
Self-Speculative Ultrametric Decode (Milestone 3.2).

Uses a single surgical model to:
  1. Draft `gamma` tokens autoregressively using aggressive ultrametric tree pruning
     (`draft_req_depth`, e.g., 4 — ~90% KV cache sparsity).
  2. Verify the `gamma` drafted tokens in a single parallel forward pass using dense
     or shallow-depth attention (`verify_req_depth = 0`).
  3. Apply exact speculative rejection sampling (Leviathan et al., 2023; Chen et al., 2023)
     so the output distribution is mathematically identical to the dense verifier model!
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class SpeculativeSurgicalGenerator:
    """
    Self-Speculative Decoding engine for SurgicalLlamaAttention models.
    Requires zero extra VRAM because the draft model IS the surgical model run at
    `surgical_req_depth = draft_req_depth` and the verifier is the same model run at
    `surgical_req_depth = verify_req_depth` (default 0 = dense).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        gamma: int = 4,
        draft_req_depth: int = 4,
        verify_req_depth: int = 0,
    ):
        self.model = model
        self.gamma = max(1, int(gamma))
        self.draft_req_depth = int(draft_req_depth)
        self.verify_req_depth = int(verify_req_depth)
        self.total_drafted = 0
        self.total_accepted = 0
        self.total_steps = 0

    def _set_req_depth(self, req_depth: Optional[int]):
        setattr(self.model.config, "surgical_req_depth", req_depth)
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is not None:
            setattr(text_cfg, "surgical_req_depth", req_depth)

    def reset_stats(self):
        self.total_drafted = 0
        self.total_accepted = 0
        self.total_steps = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return float(self.total_accepted) / float(self.total_drafted)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Generates up to `max_new_tokens` using self-speculative ultrametric decoding.
        Supports greedy (`temperature <= 0`) and stochastic (`temperature > 0`) exact
        speculative rejection sampling.
        """
        assert input_ids.shape[0] == 1, "SpeculativeSurgicalGenerator currently supports batch_size=1"
        was_training = self.model.training
        self.model.eval()
        orig_req_depth = getattr(self.model.config, "surgical_req_depth", None)

        seq = input_ids.clone()
        prompt_len = seq.shape[1]
        target_len = prompt_len + max_new_tokens

        try:
            while seq.shape[1] < target_len:
                remaining = target_len - seq.shape[1]
                cur_gamma = min(self.gamma, remaining)
                self.total_steps += 1

                # 1. Draft `cur_gamma` tokens using sparse ultrametric attention
                self._set_req_depth(self.draft_req_depth)
                draft_seq = seq.clone()
                draft_tokens = []
                draft_probs_list = []

                for _ in range(cur_gamma):
                    out = self.model(input_ids=draft_seq, use_cache=False)
                    next_logits = out.logits[:, -1, :].float()
                    if temperature <= 0.0:
                        next_tok = next_logits.argmax(dim=-1, keepdim=True)
                        probs = F.one_hot(next_tok.squeeze(-1), num_classes=next_logits.shape[-1]).float()
                    else:
                        probs = F.softmax(next_logits / temperature, dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1)
                    draft_tokens.append(next_tok)
                    draft_probs_list.append(probs)
                    draft_seq = torch.cat([draft_seq, next_tok], dim=-1)

                # 2. Verify all `cur_gamma` drafted tokens in parallel with dense verifier
                self._set_req_depth(self.verify_req_depth)
                verify_out = self.model(input_ids=draft_seq, use_cache=False)
                # Positions [seq.shape[1]-1 ... seq.shape[1]-1 + cur_gamma] predict the cur_gamma draft tokens + 1 bonus token
                start_pos = seq.shape[1] - 1
                verify_logits = verify_out.logits[:, start_pos : start_pos + cur_gamma + 1, :].float()

                self.total_drafted += cur_gamma
                n_accepted = 0
                bonus_token = None

                for i in range(cur_gamma):
                    tok_i = draft_tokens[i]  # (1, 1)
                    tok_idx = int(tok_i.item())
                    v_logits_i = verify_logits[:, i, :]

                    if temperature <= 0.0:
                        v_tok = v_logits_i.argmax(dim=-1, keepdim=True)
                        if int(v_tok.item()) == tok_idx:
                            seq = torch.cat([seq, tok_i], dim=-1)
                            n_accepted += 1
                            if eos_token_id is not None and tok_idx == eos_token_id:
                                break
                        else:
                            # Reject and replace with verifier's greedy choice
                            bonus_token = v_tok
                            break
                    else:
                        p_verify = F.softmax(v_logits_i / temperature, dim=-1)
                        q_draft = draft_probs_list[i]
                        p_val = p_verify[0, tok_idx].item()
                        q_val = max(q_draft[0, tok_idx].item(), 1e-12)
                        accept_prob = min(1.0, p_val / q_val)

                        u = torch.rand((), device=seq.device).item()
                        if u <= accept_prob:
                            seq = torch.cat([seq, tok_i], dim=-1)
                            n_accepted += 1
                            if eos_token_id is not None and tok_idx == eos_token_id:
                                break
                        else:
                            # Sample from residual distribution norm(max(0, p_verify - q_draft))
                            residual = torch.clamp(p_verify - q_draft, min=0.0)
                            res_sum = residual.sum(dim=-1, keepdim=True)
                            if res_sum.item() > 1e-12:
                                residual = residual / res_sum
                            else:
                                residual = p_verify
                            bonus_token = torch.multinomial(residual, num_samples=1)
                            break

                self.total_accepted += n_accepted

                if eos_token_id is not None and int(seq[0, -1].item()) == eos_token_id:
                    break

                if seq.shape[1] < target_len:
                    if bonus_token is not None:
                        seq = torch.cat([seq, bonus_token], dim=-1)
                    elif n_accepted == cur_gamma:
                        # All draft tokens were accepted -> sample 1 bonus token from the final verifier position
                        last_v_logits = verify_logits[:, cur_gamma, :]
                        if temperature <= 0.0:
                            bonus_token = last_v_logits.argmax(dim=-1, keepdim=True)
                        else:
                            p_last = F.softmax(last_v_logits / temperature, dim=-1)
                            bonus_token = torch.multinomial(p_last, num_samples=1)
                        seq = torch.cat([seq, bonus_token], dim=-1)

                    if eos_token_id is not None and int(seq[0, -1].item()) == eos_token_id:
                        break
        finally:
            self._set_req_depth(orig_req_depth)
            if was_training:
                self.model.train()

        if return_stats:
            total_gen = max(0, int(seq.shape[1] - prompt_len))
            stats = {
                "total_drafted": self.total_drafted,
                "total_accepted": self.total_accepted,
                "total_steps": self.total_steps,
                "acceptance_rate": self.acceptance_rate,
                "tokens_per_step": float(total_gen) / float(max(1, self.total_steps)),
            }
            return seq, stats
        return seq
