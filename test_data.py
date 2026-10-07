# old file - not in use

from loader.data_loaders import ProductDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# Setup
processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
# Make sure 'data/images' exists if your images are inside that folder 
dataset = ProductDataset(
    csv_file='data/all_products_partial.csv',  # Point to the real file name
    image_dir='data/images',                  # Ensure this folder exists
    processor=processor
)

# Test
loader = DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))

print("Successfully loaded batch. Shapes:")
for key, value in batch.items():
    print(f"{key}: {value.shape}")
