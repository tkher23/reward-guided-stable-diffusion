from __future__ import annotations

from pathlib import Path

import numpy as np
import deeplake
from PIL import Image


def download_zappos_dataset(output_dir: str = "sneakerdata/images", num_images: int = 100) -> None:
    """
    Load UT Zappos50K dataset from Deep Lake and save shoe images locally.

    This uses Activeloop Deep Lake to stream the dataset without downloading the full zip.

    Args:
        output_dir: where to save the filtered shoe images
        num_images: number of shoe images to extract (default 100)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading UT Zappos50K dataset from Deep Lake...")
    print("(This streams the data, no huge download needed)")

    try:
        # Load dataset from Deep Lake
        ds = deeplake.load("hub://activeloop/utzappo")
        print(f"Dataset loaded! Total samples: {len(ds)}")

        # Inspect structure
        print("\nInspecting dataset structure...")
        print(f"Available tensors: {ds.tensors}")

        shoes_dir = output_path / "shoes_100"
        shoes_dir.mkdir(exist_ok=True)

        # Stream images and save
        count = 0
        errors = 0

        print(f"\nExtracting {num_images} shoe images...")

        # Iterate through dataset
        for i in range(min(len(ds), num_images * 5)):  # Try 5x more samples to get 100 good ones
            if count >= num_images:
                break

            try:
                # Get image from dataset (tensor is called 'images', not 'image')
                image_sample = ds.images[i]

                # Convert to numpy
                if hasattr(image_sample, "numpy"):
                    image_array = image_sample.numpy()
                else:
                    image_array = np.array(image_sample)

                # Ensure it's uint8
                if image_array.size == 0:
                    continue

                if image_array.dtype != np.uint8:
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    else:
                        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

                # Handle different shapes
                if len(image_array.shape) == 2:  # Grayscale
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif image_array.shape[2] == 4:  # RGBA
                    image_array = image_array[:, :, :3]  # Drop alpha

                # Create PIL Image
                image = Image.fromarray(image_array)

                # Save
                output_file = shoes_dir / f"shoe_{count:03d}.jpg"
                image.save(output_file, quality=95)

                count += 1

                if count % 10 == 0:
                    print(f"  ✓ Saved {count}/{num_images} images")

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Note: Skipped sample {i} (size/format issue)")
                continue

        print(f"\n{'='*60}")
        print(f"✓ Success! Saved {count} shoe images to {shoes_dir}")
        print(f"✓ Total images: {len(list(shoes_dir.glob('*.jpg')))}")
        if errors > 3:
            print(f"  (Processed {count + errors} samples, skipped {errors})")
        print(f"{'='*60}")

        if count == 0:
            print("\nWARNING: No images were saved. Check Deep Lake connectivity.")

    except ImportError:
        print("ERROR: deeplake not installed. Install with:")
        print("  pip install deeplake")
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    download_zappos_dataset()


