import torch
import os
from PIL import Image
from utils import factory
from torchvision import transforms
import json
import argparse


def build_transform():
    input_size = 224
    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3)
        )  # to maintain the same ratio w.r.t. 224 images
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return t


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple pre-trained incremental learning algorthms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/simplecil.json",
        help="Json file of settings.",
    )
    return parser


def predict_on_images(folder_path, model, transform):
    """Predict on a folder of images using the given model."""
    predictions = {}

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model._network(
                image_tensor
            )  # assuming the model structure you've provided
            _, predicted = outputs.max(1)
            predictions[image_name] = predicted.item()

    return predictions


def main(args):
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    transform = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=3
            ),  # to maintain the same ratio w.r.t. 224 images
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    checkpoints_dir = "./outputs/{}/{}_{}_{}".format(
        args["config_id"], args["model_name"], args["dataset"], args["increment"]
    )
    for i in range(10):  # Assuming 10 tasks
        checkpoint_filename = "{}/task_{}.pkl".format(checkpoints_dir, i)
        checkpoint = torch.load(checkpoint_filename)

        model = factory.get_model(args["model_name"], args)
        model._network.load_state_dict(checkpoint["model_state_dict"])
        model._network.eval()

        predictions = predict_on_images(args.test_images_dir, model, transform)

        with open(f"result_{i}.txt", "w") as f:
            for image_name, pred in predictions.items():
                f.write(f"{image_name} {pred}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple pre-trained incremental learning algorthms."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Json file of settings.",
    )
    parser.add_argument(
        "--config_id",
        type=str,
        required=True,
        help="Identifier of the experiment.",
    )
    parser.add_argument(
        "--test_images_dir",
        type=str,
        default="/workspace/study/pilot/data/cs701/raw/val",
        help="Directory containing test images.",
    )
    args = parser.parse_args()
    main(args)
