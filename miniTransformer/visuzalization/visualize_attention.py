import os
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
import numpy as np
import torch

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_attention(input_tensors,
                        tensor_names,
                        num_heatmaps=16,
                        iter_num=0,
                        output_dir="heatmaps"):
    # Create a single row with num_heatmaps columns of subplots
    fig, axes = plt.subplots(1, num_heatmaps, figsize=(2 * num_heatmaps, 4))

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through the input_tensors and tensor_names
    for i, (input_tensor,
            tensor_name) in enumerate(zip(input_tensors, tensor_names)):
        # Detach the input tensor from the computation graph and move it to CPU
        input_tensor_np_list = [x.detach().cpu().numpy() for x in input_tensor]

        print(f"Tensor for {tensor_name} shaped {np.array(input_tensor_np_list).shape}")

        # Iterate through the list of input tensors
        for j in range(np.array(input_tensor_np_list).shape[0]):  #(num_heatmaps):
            # Plot the heatmap using the data from the input array
            sns.heatmap(input_tensor_np_list[j],
                        annot=False,
                        ax=axes[j],
                        cbar=False)

            # Set the title for the current subplot
            axes[j].set_title(f"Heatmap {j + 1} {tensor_name}")

        # Adjust the layout
        plt.tight_layout()

        # Save the heatmap to an image file
        output_path = os.path.join(output_dir,
                                   f"{tensor_name}_iter_{iter_num}.png")
        plt.savefig(output_path)

        # Clear the figure to prevent overlapping plots
        plt.clf()


def create_animation(
    tensor_names,
    num_iterations,
    heatmap_interval,
    output_dir="heatmaps",
    animation_dir="animations",
):
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)

    for tensor_name in tensor_names:
        images = []

        for i in range(0, num_iterations, heatmap_interval):
            image_path = os.path.join(output_dir, f"{tensor_name}_iter_{i}.png")
            images.append(imageio.imread(image_path))

        output_path = os.path.join(animation_dir, f"{tensor_name}_animation.gif")
        imageio.mimsave(output_path, images, fps=2)
