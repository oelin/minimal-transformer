def sample_top_k(
    model: nn.Module,
    prefix: torch.Tensor,
    sequence_length: int,
    context_length: int,
    number_of_samples: int,
    k: int = 5,
) -> torch.Tensor:
    """Perform top-k sampling."""

    assert len(prefix) > 0

    sample = torch.zeros((number_of_samples, sequence_length)).to(int)
    sample_indices = torch.arange(number_of_samples)


    prefix_length = prefix.size(0)
    sample[:, : prefix_length]  = prefix  # Impute prefix.

    for i in range(prefix_length, sequence_length):

        context = sample[:, max(0, i - context_length) : i]  # Truncate context. 
        mask = convert_mask_from_binary_to_float(get_causal_mask(context))

        logits = model(context, mask)[:, -1]  # Logits for next token.
        logits = torch.topk(logits, k=k)  # Logits restricted to top-k highest.

        logit_indices = torch.multinomial(F.softmax(logits.values, dim=1), num_samples=1, replacement=True)
        token_indices = logits.indices[sample_indices, logit_indices.squeeze(1)]  # Next token inddex for each sample.

        sample[:, i] = token_indices  # Impute next tokens.
    
    return sample
