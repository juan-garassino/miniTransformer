import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
import numpy as np
import os
import glob

def save_plot(output_dir, tensor_name, iter_num):
    """
    Save the current plot to the specified output directory.

    :param output_dir: Directory to save the plot
    :param tensor_name: Name of the tensor to include in the filename
    :param iter_num: Iteration number to include in the filename
    """
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{tensor_name}_iter_{iter_num}.png")
    plt.savefig(output_path)
    plt.clf()

    print(f"\n✅ {Fore.GREEN}Plot saved to {output_path}{Style.RESET_ALL}")

def create_animation(tensor_names, output_dir="heatmaps", animation_dir="animations"):
    """
    Create and save animations from heatmap images after the last iteration.

    :param tensor_names: List of tensor names for creating animations.
    :param output_dir: Directory containing heatmap images.
    :param animation_dir: Directory to save the animations.
    """

    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)

    for tensor_name in tensor_names:
        images = []

        # Get the list of files in the output directory for the tensor
        file_pattern = os.path.join(output_dir, f"{tensor_name}_iter_*.png")
        file_list = glob.glob(file_pattern)
        # Ensure the files are sorted by iteration number
        file_list.sort(key=lambda x: int(os.path.basename(x).split('_iter_')[-1].split('.')[0]))

        for file_path in file_list:
            images.append(imageio.imread(file_path))

        output_path = os.path.join(animation_dir, f"{tensor_name}_animation.gif")
        # Here, duration is the time spent on each frame in seconds.
        imageio.mimsave(output_path, images, duration=0.2)

        print(f"\n🎦 {Fore.BLUE}Animation saved to {output_path}{Style.RESET_ALL}")

def plot_heatmaps(axes, input_tensor_np, tensor_name, grid_size, font_size, cmap):
    """
    Plot heatmaps for the input tensor in the specified axes, displaying only the title and the heatmaps.

    :param axes: Axes for the subplots.
    :param input_tensor_np: Input tensor as a NumPy array.
    :param tensor_name: Name of the tensor to display in the title of each heatmap.
    :param grid_size: Tuple representing the grid size (num_layers, num_heads).
    :param font_size: Font size for the heatmap titles.
    :param cmap: Colormap for the heatmaps.
    """
    num_layers, num_heads = grid_size
    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer, head]
            sns.heatmap(
                input_tensor_np[layer, head, :, :],  # Adjust indexing for the new tensor shape
                annot=False,
                ax=ax,
                cbar=False,
                cmap=cmap,
            )
            ax.set_title(f"{tensor_name} Layer {layer + 1} Head {head + 1}", fontsize=font_size)
            
            # Remove tick marks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            # Also remove axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')

def visualize_attention(input_tensors, tensor_names, iter_num=0, output_dir="heatmaps",
                        animation_dir="animations", animation=False, heatmap_interval=1,
                        resolution=250, font_size=5, cmap="viridis", grid_size=None):
    """
    Visualize attention heatmaps for the input tensors.

    :param input_tensors: List of input tensors to visualize, structured as [layer, head, matrix].
    :param tensor_names: List of names for the input tensors.
    :param iter_num: Iteration number for the current visualization.
    :param output_dir: Directory to save the heatmap images.
    :param animation_dir: Directory to save the animations.
    :param animation: Whether to create animations.
    :param heatmap_interval: Interval between frames in the animation.
    :param resolution: Resolution (dpi) of the saved plots.
    :param font_size: Font size for the plot titles.
    :param cmap: Colormap for the heatmaps.
    :param grid_size: Tuple representing the grid size (num_layers, num_heads).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_tensor, tensor_name in zip(input_tensors, tensor_names):
        input_tensor_np = np.array(input_tensor.detach().cpu().numpy())

        fig, axes = plt.subplots(*grid_size, figsize=(2 * grid_size[1], 2 * grid_size[0]), dpi=resolution)

        plot_heatmaps(axes, input_tensor_np, tensor_name, grid_size, font_size, cmap)

        fig_path = os.path.join(output_dir, f"{tensor_name}_iter_{iter_num:05}.png")
        plt.savefig(fig_path)
        plt.close(fig)

        print(f"\n🆕 Created heatmaps for {tensor_name} at {fig_path}")

