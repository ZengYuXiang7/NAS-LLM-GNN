# coding : utf-8
# Author : yuxiang Zeng

import torch
import numpy as np

# 精度计算
def ErrorMetrics(realVec, estiVec):
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, torch.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)
    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, torch.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)

    absError = np.abs(estiVec - realVec)
    MAE = np.mean(absError)
    RMSE = np.linalg.norm(absError) / np.sqrt(np.array(absError.shape[0]))
    NMAE = np.sum(np.abs(realVec - estiVec)) / np.sum(realVec)
    NRMSE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / np.sqrt(np.sum(realVec ** 2))

    Acc = []
    thresholds = [0.01, 0.05, 0.10]
    for threshold in thresholds:
        threshold = realVec * threshold  # 定义阈值为真实值的5%
        accurate_predictions = absError < threshold
        accuracy = np.mean(accurate_predictions.astype(float))
        Acc.append(accuracy)

    return {
        'MAE' : MAE,
        'RMSE' : RMSE,
        'NMAE': NMAE,
        'NRMSE': NRMSE,
        'Acc' : Acc,
    }
