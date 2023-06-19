import math
import os
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
import numpy as np
import os
import glob


def plot_heatmaps(
    axes, input_tensor_np, tensor_name, num_heatmaps, grid_size, font_size, cmap
):
    """
    Plot heatmaps for the input tensor in the specified axes.

    :param axes: Axes for the subplots
    :param input_tensor_np: Input tensor as a NumPy array
    :param tensor_name: Name of the tensor to display in the title of each heatmap
    :param num_heatmaps: Number of heatmaps to plot
    :param grid_size: Size of the grid for subplot arrangement
    :param font_size: Font size for the heatmap titles and axis labels
    :param cmap: Colormap for the heatmaps
    """
    for j in range(num_heatmaps):
        row, col = divmod(j, grid_size)
        ax = axes[row, col]
        sns.heatmap(
            input_tensor_np[:, j, :],
            annot=False,
            ax=ax,
            cbar=False,
            cmap=cmap,
            annot_kws={"fontsize": font_size},
        )
        ax.set_title(f"Heatmap {j + 1} {tensor_name}", fontsize=font_size)
        ax.tick_params(axis="both", which="both", labelsize=font_size)
        ax.set_xlabel("Input Sequence Position", fontsize=font_size)
        ax.set_ylabel("Attention Distribution", fontsize=font_size)


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


def create_animation(
    tensor_names,
    # heatmap_interval,
    output_dir="miniTransformer/heatmaps",
    animation_dir="miniTransformer/animations",
):
    """
    Create and save animations from heatmap images.

    :param tensor_names: List of tensor names for creating animations
    :param heatmap_interval: Interval between frames in the animation
    :param output_dir: Directory containing heatmap images
    :param animation_dir: Directory to save the animations
    """

    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)

    for tensor_name in tensor_names:
        images = []

        # Get the list of files in the output directory for the tensor
        file_pattern = os.path.join(output_dir, f"{tensor_name}_iter_*.png")
        file_list = glob.glob(file_pattern)
        file_list.sort()

        for file_path in file_list:
            images.append(imageio.imread(file_path))

        output_path = os.path.join(animation_dir, f"{tensor_name}_animation.gif")
        imageio.mimsave(output_path, images, duration=200)

        print(f"\n🎦 {Fore.BLUE}Animation saved to {output_path}{Style.RESET_ALL}")


def visualize_attention(
    input_tensors,
    tensor_names,
    iter_num=0,
    output_dir="miniTransformer/miniTransformer/heatmaps",
    animation_dir="miniTransformer/miniTransformer/animations",
    animation=True,
    heatmap_interval=1,
    resolution=250,
    font_size=5,
    cmap="viridis",
):
    """
    Visualize attention heatmaps for the input tensors.

    :param input_tensors: List of input tensors to visualize
    :param tensor_names: List of names for the input tensors
    :param iter_num: Iteration number for the current visualization
    :param output_dir: Directory to save the heatmap images
    :param animation_dir: Directory to save the animations
    :param num_iterations: Number of iterations for the animation
    :param heatmap_interval: Interval between frames in the animation
    :param resolution: Resolution (dpi) of the saved plots
    :param font_size: Font size for the plot titles
    :param cmap: Colormap for the heatmaps
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (input_tensor, tensor_name) in enumerate(zip(input_tensors, tensor_names)):
        input_tensor_np = np.array([x.detach().cpu().numpy() for x in input_tensor])
        num_heatmaps = input_tensor_np.shape[1]

        grid_size = int(math.ceil(math.sqrt(num_heatmaps)))
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(2 * grid_size, 6), dpi=resolution
        )

        plot_heatmaps(
            axes, input_tensor_np, tensor_name, num_heatmaps, grid_size, font_size, cmap
        )

        save_plot(output_dir, tensor_name, iter_num)

        print(f"\n🆕 {Fore.YELLOW}Created heatmaps for {tensor_name}{Style.RESET_ALL}")

        plt.close(fig)

    if animation:
        create_animation(
            tensor_names,
            # heatmap_interval,
            output_dir="/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/heatmaps",
            animation_dir="/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/animations",
        )
