def get_causal_mask(x: torch.Tensor) -> torch.Tensor:
  
    sequence_length = x.size(1)
    mask = (1 - torch.tril(torch.ones((sequence_length, sequence_length)))).float()

    return mask


def get_padding_mask(x: torch.Tensor, maximum_sequence_length: int) -> torch.Tensor:

    sequence_length = x.size(1)
    mask = torch.ones((maximum_sequence_length, maximum_sequence_length))
    mask[:, : sequence_length] = 0.

    return mask

def convert_mask_from_binary_to_float(mask: torch.Tensor) -> torch.Tensor:

    mask.masked_fill_(mask.bool(), -torch.inf)

    return mask
