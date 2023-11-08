import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import numpy as np
import shutil
from utils import calculate_residual_count_average,load_model,image_preprocess,calculate_residual
import statistics

TEST_DIR = "TA-Class8/Test/NonDefect"
# TEST_DIR = "Test-Defect"
# TEST_DIR = "./image"
DEFECT_ANS_FILE = "Defect.txt" #  record the defect images
threshold = 0.007

# a = [44, 52, 53, 61, 63, 67, 67, 68, 71, 72, 73, 74, 75, 75, 75, 77, 78, 79, 79, 81, 82, 84, 84, 85, 85, 86, 87, 88, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 97, 97, 98, 98, 100, 100, 102, 102, 103, 103, 104, 104, 105, 105, 106, 107, 108, 108, 108, 108, 113, 113, 113, 114, 115, 115, 115, 116, 116, 117, 117, 118, 118, 118, 119, 119, 120, 120, 121, 122, 123, 124, 130, 133, 135, 135, 137, 139, 139, 140, 140, 141, 141, 142, 142, 144, 146, 147, 148, 148, 149, 149, 150, 151, 151, 153, 154, 154, 154, 155, 155, 156, 162, 163, 163, 164, 164, 169, 170, 171, 171, 171, 172, 172, 173, 174, 176, 176, 179, 180, 187, 193, 195, 195, 196, 199, 200, 201, 203, 203, 206, 208, 209, 211, 213, 216, 222, 224, 246, 262, 269, 297]
# print(statistics.median(list(set(a))))
# exit()
# 計算所有有缺陷的 image 的 residual > threshold 的 element  (當 threshold = 0.007 算出來 average是 132, median value 是 120)
# average_residual_count = calculate_residual_count_average(threshold=threshold)
# print(average_residual_count)
# exit()

# load model
model = load_model()

y_test = []
y_pred = []

residual_count_threshold = 132
# read DEFECT_ANS_FILE
with open(DEFECT_ANS_FILE) as f:
    defect_lines = f.read().splitlines()
    
# list TEST_DIR images and calculate residual
for image in os.listdir(TEST_DIR):
    print(f'Image: {image}')
    if not image.endswith('.PNG'):
        continue
    imgpath = os.path.join(TEST_DIR,image)
    img = image_preprocess(imgpath)
    print(img)
    y = model(img)
    print(y)
    np_residual = calculate_residual(img,y)
    count = np.count_nonzero(np_residual > threshold)
    print(np_residual > threshold)
    print(f'======residual elements count: {count}=====')

    if image.split('.')[0] in defect_lines:
        y_test.append(1)
    else:
        y_test.append(0)
        
    is_defect = 1 if count > residual_count_threshold else 0
    y_pred.append(is_defect)

cm=confusion_matrix(y_test,y_pred)    
print(cm)

# draw the confusion_matrix
plt.figure(figsize=(15,8))
sns.heatmap(cm,square=True,annot=True,fmt='d',linecolor='white',cmap='RdBu',linewidths=1.5,cbar=False)
plt.xlabel('Pred',fontsize=20)
plt.ylabel('True',fontsize=20)
plt.show()

print(classification_report(y_test,y_pred))