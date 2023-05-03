import seaborn as sns
import matplotlib.pyplot as plt


def visualize_attention(input_tensors, tensor_names, num_heatmaps=16):
    """
    Visualize the attention tensors as heatmaps.

    Args:
        input_tensors (list): A list of tensors (keys, queries, values) to visualize.
        tensor_names (list): A list of tensor names corresponding to the input_tensors.
        num_heatmaps (int): The number of heatmaps to display per tensor.
    """
    # Create a single row with num_heatmaps columns of subplots
    fig, axes = plt.subplots(1, num_heatmaps, figsize=(2 * num_heatmaps, 4))

    # Iterate through the input_tensors and tensor_names
    for input_tensor, tensor_name in zip(input_tensors, tensor_names):
        # Convert the input tensor to a NumPy array
        input_tensor_np = input_tensor.cpu().numpy()

        for i in range(num_heatmaps):
            # Plot the heatmap using the data from the input array
            sns.heatmap(input_tensor_np[i], annot=False, ax=axes[i], cbar=False)

            # Set the title for the current subplot
            axes[i].set_title(f"Heatmap {i + 1} {tensor_name}")

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()
