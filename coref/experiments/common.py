import einops
import torch


def lookup_answer_logits(logits, answer_tokens, normalize=True, query_position=None):
    """
    logits: (batch_size, seq_len, vocab_size)
    answer_tokens: (batch_size, num_answers)
    query_position: None | (batch_size)

    """
    # Only the final logits are relevant for the answer
    device = logits.device
    if query_position is None:
        final_logits = logits[:, -1, :]
    else:
        final_logits = logits.gather(
            dim=1,
            index=einops.repeat(
                query_position, "batch -> batch dummy dim", dummy=1, dim=logits.shape[2]
            ).to(device),
        )[:, 0, :]
        assert final_logits.shape == logits[:, -1, :].shape
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens.to(device))
    if normalize:
        answer_logits = answer_logits - final_logits.logsumexp(dim=-1, keepdim=True)
    return answer_logits


def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    # will take [:, 0] - [:, 1]
    answer_logits = lookup_answer_logits(
        logits.cpu(), answer_tokens.cpu(), normalize=False
    )
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()


from rich import print as rprint


def pretty_print_logits(tokenizer, logits):
    top_k = 10
    token_probs = logits.softmax(-1)
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
    # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
    # rprint gives rich text printing
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{tokenizer.decode(sorted_token_values[i])}|"
        )


def repeat(num, fn):
    return torch.stack([fn(i) for i in range(num)])
