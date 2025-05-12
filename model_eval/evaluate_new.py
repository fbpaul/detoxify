import json
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import ToxicClassifier
from src.data_loaders import JigsawDataBias, JigsawDataMultilingual, JigsawDataOriginal
import src.data_loaders as module_data

def get_instance(module, name, config, *args, **kwargs):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

def test_classifier(config, dataset, checkpoint_path, device="cuda:1"):
    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    config["dataset"]["args"]["test_csv_file"] = dataset

    test_dataset = get_instance(module_data, "dataset", config, train=False)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=20,
        shuffle=False,
    )

    scores = []
    targets = []
    ids = []
    for *items, meta in tqdm(test_data_loader):
        if "multi_target" in meta:
            targets += meta["multi_target"]
        else:
            targets += meta["target"]

        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        scores.extend(sm)

    binary_scores = [s >= 0.5 for s in scores]
    binary_scores = np.stack(binary_scores)
    scores = np.stack(scores)
    targets = np.stack(targets)
    auc_scores = []

    for class_idx in range(scores.shape[1]):
        mask = targets[:, class_idx] != -1
        target_binary = targets[mask, class_idx]
        class_scores = scores[mask, class_idx]
        try:
            auc = roc_auc_score(target_binary, class_scores)
            auc_scores.append(auc)
        except Exception:
            warnings.warn(
                "Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
            )
            auc_scores.append(np.nan)

    mean_auc = np.mean(auc_scores)
    # 檢查ids是否是張量類型，如果不是則不需要調用tolist()
    if isinstance(ids[0], torch.Tensor):
        ids = [i.tolist() for i in ids]
    else:
        ids = list(ids)
    results = {
        "scores": scores.tolist(),
        "targets": targets.tolist(),
        "auc_scores": auc_scores,
        "mean_auc": mean_auc,
        "ids": ids,
    }

    return results

# 配置文件路徑
config_path = 'configs/Toxic_comment_classification_ModernBERT.json'  # 請替換為您的配置文件實際路徑
config = json.load(open(config_path))

# 測試集路徑
test_csv_path = 'jigsaw_data/jigsaw-toxic-comment-classification-challenge/test.csv'

# 檢查點路徑
checkpoint_path = 'saved/Jigsaw_ModernBERT/lightning_logs/version_1/checkpoints/epoch=99-step=41600.ckpt'

# 設備
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 測試分類器
results = test_classifier(config, test_csv_path, checkpoint_path, device)
# print(results)
# 保存結果
# test_set_name = test_csv_path.split("/")[-2]
# file_name = checkpoint_path.split('/')[-1][:-5] + f"results_{test_set_name}.json"
# with open(file_name, "w") as f:
#     json.dump(results, f)
# print(f"Saving results to {file_name}")

# 設定threshold
threshold = 0.5
# 轉換scores為預測標籤
predicted_labels = np.array(results['scores']) > threshold
# 將targets轉換為numpy array
true_labels = np.array(results['targets'])
# 計算指標
report = classification_report(true_labels, predicted_labels, target_names=[f'Class {i}' for i in range(6)])
print(report)