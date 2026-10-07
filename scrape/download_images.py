#!/usr/bin/env python3
# Run: python scrape/download_images.py  (downloads images + writes training CSV)
"""
Download product images from CSV using Selenium and build a training-ready CSV.

Usage:
    # Test mode (3 images):
    python scrape/download_images.py --limit 3

    # Full download:
    python scrape/download_images.py

    # Re-download existing files:
    python scrape/download_images.py --force

    # Filter by category:
    python scrape/download_images.py --category bracelets --limit 5
"""

import argparse
import base64
import csv
import os
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_PATH = PROJECT_ROOT / "data" / "all_products_partial.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "catalog_clip_ready.csv"

# Settings
DELAY_BETWEEN_DOWNLOADS = 0.3
PAGE_LOAD_WAIT = 2.0
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
OUTPUT_COLUMNS = ["id", "image_path", "name", "elements", "why_love", "dimensions", "price"]


def get_first_value(row: dict, keys: list[str], default: str = "") -> str:
    """Return the first non-empty field from a row for any of the candidate keys."""
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def get_extension_from_url(url: str) -> str:
    """Extract image extension from URL, with a safe fallback."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    ext = os.path.splitext(path)[1].lower()
    return ext if ext in VALID_EXTENSIONS else ".png"


class ImageDownloader:
    """Download images through a browser session to reduce HTTP blocking."""

    def __init__(self, headless: bool = True):
        self.driver = None
        self.headless = headless

    def start(self):
        """Initialize the Chrome browser session."""
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=800,600")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 "
            "Safari/537.36"
        )

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

        print("Initializing browser session...")
        self.driver.get("https://us.pandora.net/")
        time.sleep(PAGE_LOAD_WAIT)
        print("Browser ready.")

    def stop(self):
        """Close the browser if it is running."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def download_image(self, url: str, save_path: Path) -> tuple[bool, str]:
        """Download image bytes via in-page JS fetch and save to disk."""
        try:
            script = """
            async function fetchImage(url) {
                try {
                    const response = await fetch(url, {
                        method: 'GET',
                        credentials: 'include',
                        headers: {
                            'Accept': 'image/*'
                        }
                    });
                    if (!response.ok) {
                        return {error: 'HTTP ' + response.status};
                    }
                    const blob = await response.blob();
                    return new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve({data: reader.result, size: blob.size});
                        reader.onerror = reject;
                        reader.readAsDataURL(blob);
                    });
                } catch (e) {
                    return {error: e.message};
                }
            }
            return await fetchImage(arguments[0]);
            """

            result = self.driver.execute_script(script, url)
            if not result:
                return False, "No response"
            if "error" in result:
                return False, str(result["error"])

            data_url = result.get("data", "")
            if not data_url or "," not in data_url:
                return False, "Invalid data URL"

            base64_data = data_url.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
            if len(image_bytes) < 1024:
                return False, f"Too small: {len(image_bytes)}B"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as file_handle:
                file_handle.write(image_bytes)

            return True, f"{len(image_bytes) // 1024}KB"
        except Exception as err:  # pragma: no cover - defensive fallback
            return False, str(err)[:80]


def build_training_row(product: dict, image_id: str, image_rel_path: str) -> dict:
    """Create one training row matching README input expectations."""
    return {
        "id": image_id,
        "image_path": image_rel_path,
        "name": get_first_value(product, ["name", "Name", "product_name", "Product Name"]),
        "elements": get_first_value(product, ["elements", "Elements"]),
        "why_love": get_first_value(
            product,
            ["why_love", "Why Love", "Why you'll love it", "Why You'll Love It"],
        ),
        "dimensions": get_first_value(product, ["dimensions", "Dimensions"]),
        "price": get_first_value(product, ["price", "Price"]),
    }


def write_training_csv(rows: list[dict], output_csv: Path) -> None:
    """Write the CLIP training CSV with stable columns."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def download_images(
    products: list[dict],
    downloader: ImageDownloader,
    image_dir: Path,
    force: bool,
    delay_seconds: float,
) -> tuple[list[dict], dict, list[tuple[str, str, str]]]:
    """Download images and return training rows + summary stats + failures."""
    downloaded = 0
    skipped = 0
    failed = 0
    failed_products: list[tuple[str, str, str]] = []
    training_rows: list[dict] = []

    for index, product in enumerate(products, 1):
        image_id = get_first_value(product, ["sku", "product_id", "Product ID"], f"unknown_{index}")
        jewelry_type = get_first_value(product, ["jewelry_type", "Jewelry Type"], "other")
        jewelry_type_slug = jewelry_type.lower().replace(" ", "_")
        image_url = get_first_value(product, ["primary_image", "Primary Image"])

        ext = get_extension_from_url(image_url)
        image_rel_path = f"{jewelry_type_slug}/{image_id}{ext}"
        save_path = image_dir / image_rel_path

        if save_path.exists() and not force:
            print(f"[{index}/{len(products)}] [SKIP] {image_rel_path} (exists)")
            skipped += 1
            training_rows.append(build_training_row(product, image_id, image_rel_path))
            continue

        success, message = downloader.download_image(image_url, save_path)
        if success:
            print(f"[{index}/{len(products)}] [OK]   {image_rel_path} ({message})")
            downloaded += 1
            training_rows.append(build_training_row(product, image_id, image_rel_path))
        else:
            print(f"[{index}/{len(products)}] [FAIL] {image_rel_path} - {message}")
            failed += 1
            failed_products.append((image_id, image_url, message))

        if index < len(products):
            time.sleep(delay_seconds)

    stats = {"downloaded": downloaded, "skipped": skipped, "failed": failed, "total": len(products)}
    return training_rows, stats, failed_products


def main() -> None:
    parser = argparse.ArgumentParser(description="Download product images from CSV")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit number of images to download (0 = all).",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-download even if file already exists",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default=None,
        help="Filter by jewelry type (e.g., bracelets, charms)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(CSV_PATH),
        help=f"Path to CSV file (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--visible",
        "-v",
        action="store_true",
        help="Show browser window (useful for debugging)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(OUTPUT_CSV_PATH),
        help=f"Path to generated training CSV (default: {OUTPUT_CSV_PATH})",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv)
    output_csv_path = Path(args.output_csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Reading CSV: {csv_path}")
    products = []
    with open(csv_path, "r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            jewelry_type = get_first_value(row, ["jewelry_type", "Jewelry Type"], "other")
            image_url = get_first_value(row, ["primary_image", "Primary Image"])

            if args.category and jewelry_type.lower() != args.category.lower():
                continue
            if not image_url:
                continue

            products.append(row)

    total_with_images = len(products)
    if args.limit > 0:
        products = products[: args.limit]

    print(f"Found {total_with_images} products with images")
    if args.category:
        print(f"Filtered to category: {args.category}")
    if args.limit > 0:
        print(f"Limited to: {args.limit} images")
    print(f"Output directory: {IMAGE_DIR}")
    print(f"Training CSV output: {output_csv_path}")
    print("-" * 50)

    downloader = ImageDownloader(headless=not args.visible)
    downloader.start()

    try:
        training_rows, stats, failed_products = download_images(
            products=products,
            downloader=downloader,
            image_dir=IMAGE_DIR,
            force=args.force,
            delay_seconds=DELAY_BETWEEN_DOWNLOADS,
        )
    finally:
        downloader.stop()

    write_training_csv(training_rows, output_csv_path)

    print("-" * 50)
    print(
        "Done. "
        f"Downloaded: {stats['downloaded']}, "
        f"Skipped: {stats['skipped']}, "
        f"Failed: {stats['failed']}"
    )
    print(f"Wrote {len(training_rows)} training rows to: {output_csv_path}")
    if failed_products:
        print(f"\nFailed downloads ({len(failed_products)}):")
        for image_id, _, msg in failed_products[:10]:
            print(f"  - {image_id}: {msg}")
        if len(failed_products) > 10:
            print(f"  ... and {len(failed_products) - 10} more")

    print(f"\nImages saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
