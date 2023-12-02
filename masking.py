def get_causal_mask(x: torch.Tensor) -> torch.Tensor:
    sequence_length = x.size(1)

    mask = (1 - torch.tril(torch.ones((sequence_length, sequence_length))))

    return mask.bool()


def convert_mask_from_binary_to_float(mask: torch.Tensor) -> torch.Tensor:

    mask = mask.clone().float()
    mask.masked_fill_(mask.bool(), -torch.inf)

    return mask
