import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

def save_images_and_generate_file_list(dataset, folder_name, file_list_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_list_path = os.path.join(folder_name, file_list_name)
    with open(file_list_path, 'w') as file_list:
        for idx, (image, label) in enumerate(dataset):
            image_file_name = f"{idx}.png"
            image_path = os.path.join(folder_name, image_file_name)
            image.save(image_path)
            file_list.write(f"{image_file_name} {label}\n")

def main():
    # Setup paths
    dataset_root = "mnist"
    train_folder = os.path.join(dataset_root, "train")
    val_folder = os.path.join(dataset_root, "validation")

    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transforms.ToTensor())
    val_dataset = datasets.MNIST(root=dataset_root, train=False, download=True, transform=transforms.ToTensor())

    # Convert to PIL Image and save images to disk
    train_dataset = [(transforms.ToPILImage()(image), label) for image, label in train_dataset]
    val_dataset = [(transforms.ToPILImage()(image), label) for image, label in val_dataset]

    # Save images and generate file lists
    save_images_and_generate_file_list(train_dataset, train_folder, "training_file_list.txt")
    save_images_and_generate_file_list(val_dataset, val_folder, "validation_file_list.txt")

    print("MNIST dataset processing complete.")

if __name__ == "__main__":
    main()
