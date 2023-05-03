import torch
from miniTransformer.sourcing.sourcing import create_data_batch

@torch.no_grad()
def estimate_loss(model,
                  train_data,
                  val_data,
                  eval_iters,
                  block_size=None,
                  batch_size=None,
                  device=None):
    """
    Estimate the average loss for the model on training and validation data.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        eval_iters (int): The number of iterations to use for evaluation.
        get_batch (callable): A function that generates input and target batches.

    Returns:
        out (dict): A dictionary containing the average losses for training and validation data.
    """
    # Initialize the output dictionary
    out = {}

    # Set the model to evaluation mode
    model.eval()

    # Loop through both training and validation data splits
    for split in ['train', 'val']:
        # Initialize a tensor to store loss values
        losses = torch.zeros(eval_iters)

        # Loop through the specified number of evaluation iterations
        for k in range(eval_iters):
            # Get a batch of input and target data
            X, Y = create_data_batch(train_data, val_data,
                                     split,
                                     block_size=block_size,
                                     batch_size=batch_size,
                                     device=device)

            # Calculate the logits and loss for the current batch
            logits, loss = model(X, Y)

            # Store the loss value for the current iteration
            losses[k] = loss.item()

        # Calculate the average loss for the current data split
        out[split] = losses.mean()

    # Set the model back to training mode
    model.train()

    return out
