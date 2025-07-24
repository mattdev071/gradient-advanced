import os
import shutil
import zipfile

from core import constants as cst
import torchvision.transforms as T
from PIL import Image


def is_image_file_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def augment_image(image_path, output_path):
    image = Image.open(image_path)
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=image.size[0], scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    augmented = transform(image)
    augmented.save(output_path)


def prepare_dataset(
    training_images_zip_path: str,
    training_images_repeat: int,
    instance_prompt: str,
    class_prompt: str,
    job_id: str,
    regularization_images_dir: str = None,
    regularization_images_repeat: int = None,
):
    extraction_dir = f"{cst.DIFFUSION_DATASET_DIR}/tmp/{job_id}/"
    os.makedirs(extraction_dir, exist_ok=True)
    with zipfile.ZipFile(training_images_zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

    extracted_items = [entry for entry in os.listdir(extraction_dir)]
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(extraction_dir, extracted_items[0])):
        training_images_dir = os.path.join(extraction_dir, extracted_items[0])
    else:
        training_images_dir = extraction_dir

    output_dir = f"{cst.DIFFUSION_DATASET_DIR}/{job_id}/"
    os.makedirs(output_dir, exist_ok=True)

    training_dir = os.path.join(
        output_dir,
        f"img/{training_images_repeat}_{instance_prompt} {class_prompt}",
    )

    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)

    # Remove duplicates and filter corrupt images
    seen_hashes = set()
    valid_images = []
    for file in os.listdir(training_images_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(training_images_dir, file)
            if not is_image_file_valid(file_path):
                os.remove(file_path)
                continue
            with open(file_path, "rb") as f:
                file_hash = hash(f.read())
            if file_hash in seen_hashes:
                os.remove(file_path)
                continue
            seen_hashes.add(file_hash)
            valid_images.append(file_path)
    # Apply augmentation to all valid images
    for img_path in valid_images:
        augment_image(img_path, img_path)

    shutil.copytree(training_images_dir, training_dir)

    if regularization_images_dir is not None:
        regularization_dir = os.path.join(
            output_dir,
            f"reg/{regularization_images_repeat}_{class_prompt}",
        )

        if os.path.exists(regularization_dir):
            shutil.rmtree(regularization_dir)
        shutil.copytree(regularization_images_dir, regularization_dir)

    if not os.path.exists(os.path.join(output_dir, "log")):
        os.makedirs(os.path.join(output_dir, "log"))

    if not os.path.exists(os.path.join(output_dir, "model")):
        os.makedirs(os.path.join(output_dir, "model"))

    if os.path.exists(extraction_dir):
        shutil.rmtree(extraction_dir)

    if os.path.exists(training_images_zip_path):
        os.remove(training_images_zip_path)
