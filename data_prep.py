from isic_api import ISICApi

# Download ISIC 2019 and 2020 datasets via API
def download_isic_dataset(metadata_csv_url, save_dir):
    # (Use ISIC API to fetch images and labels)
    return dataset

# Split into base (common) and incremental (rare) tasks
base_dataset = download_isic_dataset(ISIC_2019_URL, "base_data")
incremental_tasks = {
    'task1': download_isic_dataset(ISIC_2020_TASK1_URL, "task1"),
    'task2': download_isic_dataset(ISIC_2020_TASK2_URL, "task2"),
    # ...
}

# Preprocess images to 224x224 and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])