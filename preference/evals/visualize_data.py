from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def visualize_shoe_images(
    image_dir: str = "sneakerdata/images/shoes_100",
    labels_file: str = "sneakerdata/features/feature_labels.json",
    num_images: int = 20,
    output_file: str = "preference/results/shoe_samples_visualization.png",
) -> None:
    """
    Display shoe images in a grid with their CLIP feature labels.

    Args:
        image_dir: directory containing shoe images
        labels_file: path to feature_labels.json
        num_images: how many images to display (will be arranged in grid)
        output_file: where to save the visualization
    """
    image_path = Path(image_dir)
    image_files = sorted(image_path.glob("*.jpg"))[:num_images]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # Load labels if available
    labels = {}
    if Path(labels_file).exists():
        with open(labels_file) as f:
            labels = json.load(f)
        print(f"Loaded labels for {len(labels)} images")
    else:
        print(f"No labels file found at {labels_file}")
        print("(Run label_features_with_ai.py first to generate labels)")

    # Create grid
    num_cols = 5
    num_rows = (len(image_files) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 3 * num_rows))
    axes = axes.flatten()

    for idx, img_file in enumerate(image_files):
        ax = axes[idx]

        # Load and display image
        image = Image.open(img_file)
        ax.imshow(image)
        ax.axis("off")

        # Get features from labels
        img_id = img_file.stem
        if img_id in labels:
            label_data = labels[img_id]
            colorful = label_data.get("colorful", 0)
            chunky = label_data.get("chunky", 0)
            clean = label_data.get("clean", 0)

            # Create label text
            label_text = (
                f"Colorful: {colorful:.2f}\n"
                f"Chunky: {chunky:.2f}\n"
                f"Clean: {clean:.2f}"
            )
        else:
            label_text = f"{img_id}\n(no labels)"

        ax.set_title(label_text, fontsize=9, pad=5)

    # Hide unused subplots
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save and show
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    print(f"\nVisualization saved to {output_file}")
    print(f"Total images displayed: {len(image_files)}")

    # Also print summary stats
    if labels:
        print("\n" + "=" * 60)
        print("FEATURE STATISTICS (first 20 images)")
        print("=" * 60)

        colorful_vals = [labels[f.stem].get("colorful", 0) for f in image_files if f.stem in labels]
        chunky_vals = [labels[f.stem].get("chunky", 0) for f in image_files if f.stem in labels]
        clean_vals = [labels[f.stem].get("clean", 0) for f in image_files if f.stem in labels]

        import numpy as np

        print(
            f"Colorful: mean={np.mean(colorful_vals):.2f}, "
            f"min={np.min(colorful_vals):.2f}, max={np.max(colorful_vals):.2f}"
        )
        print(
            f"Chunky:   mean={np.mean(chunky_vals):.2f}, "
            f"min={np.min(chunky_vals):.2f}, max={np.max(chunky_vals):.2f}"
        )
        print(
            f"Clean:    mean={np.mean(clean_vals):.2f}, "
            f"min={np.min(clean_vals):.2f}, max={np.max(clean_vals):.2f}"
        )


def list_shoe_images(image_dir: str = "sneakerdata/images/shoes_100") -> None:
    """Print list of downloaded shoe images."""
    image_path = Path(image_dir)
    image_files = sorted(image_path.glob("*.jpg"))

    print(f"\nFound {len(image_files)} shoe images in {image_dir}:")
    print("-" * 60)

    for i, img_file in enumerate(image_files[:20]):  # Show first 20
        size_mb = img_file.stat().st_size / (1024 * 1024)
        print(f"  {i+1:3d}. {img_file.name:30s} ({size_mb:.1f} MB)")

    if len(image_files) > 20:
        print(f"  ... and {len(image_files) - 20} more")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize downloaded shoe images")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="sneakerdata/images/shoes_100",
        help="Directory containing shoe images",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="sneakerdata/features/feature_labels.json",
        help="Path to feature labels JSON",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=20,
        help="Number of images to display",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="preference/results/shoe_samples_visualization.png",
        help="Where to save visualization",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list the images without visualization",
    )

    args = parser.parse_args()

    if args.list:
        list_shoe_images(args.images_dir)
    else:
        visualize_shoe_images(
            image_dir=args.images_dir,
            labels_file=args.labels_file,
            num_images=args.num,
            output_file=args.output,
        )
