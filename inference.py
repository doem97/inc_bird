import torch
import os
from PIL import Image
from utils import factory
from torchvision import transforms
import json
import copy
import argparse
from tqdm import tqdm
from utils.data_manager import DataManager

from torch.utils.data import Dataset, DataLoader


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [
            f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        print("{}: {}".format(key, value))


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def inference(args):
    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    transform = transforms.Compose(
        [
            transforms.Resize(
                args["img_size"], interpolation=3
            ),  # to maintain the same ratio w.r.t. 224 images
            transforms.CenterCrop(args["img_size"]),
            transforms.ToTensor(),
        ]
    )

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes  # update args #doem97: nb mean number
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    dataset = ImageFolderDataset(args["test_images_dir"], transform=transform)

    for task in tqdm(range(data_manager.nb_tasks)):
        model.set_eval_model(data_manager)

        # Load model checkpoint
        checkpoint_path = "./outputs/{}/ckpts".format(args["config_id"])
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_filename = "{}/task_{}.pkl".format(checkpoint_path, task)
        checkpoint = torch.load(checkpoint_filename)
        model._network.load_state_dict(checkpoint["model_state_dict"])
        model._network.to(args["device"][0])

        cnn_output_dir = "./outputs/{}/cnn/results/".format(args["config_id"])
        if not os.path.exists(cnn_output_dir):
            os.makedirs(cnn_output_dir)

        nme_output_dir = "./outputs/{}/nme/results/".format(args["config_id"])
        if not os.path.exists(nme_output_dir):
            os.makedirs(nme_output_dir)

        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        y_pred_cnn, y_pred_nme, file_names = model.inference_only(loader)

        # Save CNN results
        paired_results_cnn = list(zip(file_names, y_pred_cnn))
        paired_results_cnn.sort(key=lambda x: x[0])
        with open(os.path.join(cnn_output_dir, f"result_{task+1}.txt"), "w") as f:
            for filename, pred in paired_results_cnn:
                f.write(f"{filename} {pred}\n")

        # Save NME results
        # paired_results_nme = list(zip(file_names, y_pred_nme))
        # paired_results_nme.sort(key=lambda x: x[0])
        # with open(os.path.join(nme_output_dir, f"result_{task+1}.txt"), "w") as f:
        #     for filename, pred in paired_results_nme:
        #         f.write(f"{filename} {pred}\n")

        model.after_task()


def main(args):
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        inference(args)


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
        default="./data/cs701/raw/val",
        help="Directory containing test images.",
    )
    args = parser.parse_args()
    main(args)
