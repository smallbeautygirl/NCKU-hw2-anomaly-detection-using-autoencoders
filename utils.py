import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, ToTensor

from model import AnomalyAE

# device will be 'cuda' if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_path: str = "./model/best_model_25_loss=-2.150636353559904e-06.pth",
):
    model = AnomalyAE()
    model.load_state_dict(
        torch.load("./model/best_model_25_loss=-2.150636353559904e-06.pth")
    )
    model.eval()
    return model.to(device)


def image_preprocess(imgpath: str) -> torch.Tensor:
    img = Image.open(imgpath).convert("L")  # convert to greyscale
    transform = Compose(
        [Grayscale(), ToTensor()]
    )  # Compose: Composes several transforms together.
    img = transform(img)
    img = img.to(device)
    return img.unsqueeze(0)


def calculate_residual(img: torch.Tensor, y: torch.Tensor):
    residual = torch.abs(img[0][0] - y[0][0])  # Type: torch.tensor
    return residual.detach().cpu().numpy()


def calculate_residual_count_average(
    defect_data_dir: str = "Test-Defect",
    model_path: str = "./tensorboard_logs_08112023_14-14/best_model_13_loss=-1.5691627678387418e-05.pth",
    threshold: float = 0.007,
) -> int:
    """計算在測試集中，所有已知有缺陷的影像與 reconstruected image 的 residual numpy array 的elements 數量 (需滿足所設定的門檻，ex: > 0.007)

    Args:
        defect_data_dir (str): ex: Test-Defect
        model_path (str): ex: ./tensorboard_logs_08112023_14-14/best_model_13_loss=-1.5691627678387418e-05.pth
        threshold (float)

    Returns:
        int:
    """
    defect_ans_residual_count_list = []
    for image in os.listdir(defect_data_dir):
        print(f"Image: {image}")
        if not image.endswith(".PNG"):
            continue
        imgpath = os.path.join(defect_data_dir, image)

        img = image_preprocess(imgpath)
        model = load_model(model_path)

        y = model(img)

        np_residual = calculate_residual(img, y)

        # count = np.sum(np_residual > threshold)
        print()
        # count = np_residual.size - np.count_nonzero(np_residual > threshold)
        count = np.count_nonzero(np_residual > threshold)
        # print(f"number of zeros: {}")
        print(f"Numpy Residual shape: {np_residual.shape}")
        print(f"======residual elements count: {count}=====")
        defect_ans_residual_count_list.append(count)
    # defect_ans_residual_count_list = sorted(defect_ans_residual_count_list)
    # print(defect_ans_residual_count_list)

    # defect_ans_residual_count_list = defect_ans_residual_count_list[5:-1]
    # print(defect_ans_residual_count_list)
    # del defect_ans_residual_count_list[0]
    # del defect_ans_residual_count_list[1]
    # del defect_ans_residual_count_list[-1]
    # del defect_ans_residual_count_list[-2]
    # 找出最大與最小值，並刪除
    # max_count = max(defect_ans_residual_count_list)
    # minimun_count = min(defect_ans_residual_count_list)
    # defect_ans_residual_count_list.remove(max_count)
    # defect_ans_residual_count_list.remove(minimun_count)
    # 算平均
    average_residual_count = sum(defect_ans_residual_count_list) / len(
        defect_ans_residual_count_list
    )
    print(average_residual_count)
    return average_residual_count

    # 算中位數
    # median_value = np.percentile(np.array(defect_ans_residual_count_list), 50)
    # print(median_value)
    return median_value
