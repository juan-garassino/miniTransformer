from your_package.visualization import generate_heatmap
import os


def test_generate_heatmap():
    # Assuming your visualization function saves a file
    generate_heatmap(data=[1, 2, 3], file_path="test_heatmap.png")
    assert os.path.exists("test_heatmap.png")
    os.remove("test_heatmap.png")  # Clean up after test
