# [HW2] 人工智慧在異常偵測的應用 - Semi-supervised Anomaly Detection using AutoEncoders

| 姓名 | 學號 |
| --- | --- | 
| 樊紹萱 | P77111037 |

## :one: Requirement Installation

:memo: file path: `requirements.txt`

```bash=
pip3 install -r requirements
```

## :two: Data Preparation

1. 使用助教已分好的 Class8:
    - :file_folder: data path: `TA-Class8`
2. \
## :three: Training

:memo: file path: `main.py`

```bash=
python main.py --train_dir TA-Class8/Train/NonDefect --val_dir TA-Class8/Test/NonDefect --epochs 50
```
1. 模型架構：
2. 參數調整：將 `epoch` 調為 **50**
3. 最終模型： :memo: file path: `./tensorboard_logs_29102023_15-00/models/best_model_46_loss=-2.1121409128356237e-07.pth`
4. 

## :four: 計算 F1-Score

:memo: file path: `calculate_f1_score.py`

1. 總共設兩種 threshold：

| Value | Description |
| --- | --- | 
| 0.007 | residual |
| 135 | 在 |

2. 