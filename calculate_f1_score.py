import os
import shutil
import statistics

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.transforms import Compose, Grayscale, ToTensor

from model import AnomalyAE
from utils import (
    calculate_residual,
    calculate_residual_count_average,
    image_preprocess,
    load_model,
)

TEST_DIR = "TA-Class8/Test/NonDefect"
# TEST_DIR = "Test-Defect"
# TEST_DIR = "./image"
DEFECT_ANS_FILE = "Defect.txt"  #  record the defect images
threshold = 0.007

# 計算所有有缺陷的 image 的 residual > threshold 的 element  (當 threshold = 0.007 算出來 average是 132, median value 是 120)
# average_residual_count = calculate_residual_count_average(threshold=threshold)
# print(average_residual_count)
# exit()

# load model
model = load_model(
    f"./tensorboard_logs_08112023_14-14/best_model_13_loss=-1.5691627678387418e-05.pth"
)

y_test = []
y_pred = []

residual_count_threshold = 133
# read DEFECT_ANS_FILE
with open(DEFECT_ANS_FILE) as f:
    defect_lines = f.read().splitlines()

# list TEST_DIR images and calculate residual
for image in os.listdir(TEST_DIR):
    print(f"Image: {image}")
    if not image.endswith(".PNG"):
        continue
    imgpath = os.path.join(TEST_DIR, image)
    img = image_preprocess(imgpath)
    print(img)
    y = model(img)
    print(y)
    np_residual = calculate_residual(img, y)
    # count = np_residual.size - np.count_nonzero(np_residual)
    count = np.count_nonzero(np_residual > threshold)
    print(np_residual > threshold)
    print(f"======residual elements count: {count}=====")

    if image.split(".")[0] in defect_lines:
        y_test.append(1)
    else:
        y_test.append(0)

    is_defect = 1 if count >= residual_count_threshold else 0
    y_pred.append(is_defect)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# draw the confusion_matrix
plt.figure(figsize=(15, 8))
sns.heatmap(
    cm,
    square=True,
    annot=True,
    fmt="d",
    linecolor="white",
    cmap="RdBu",
    linewidths=1.5,
    cbar=False,
)
plt.xlabel("Pred", fontsize=20)
plt.ylabel("True", fontsize=20)
plt.show()

print(classification_report(y_test, y_pred))
