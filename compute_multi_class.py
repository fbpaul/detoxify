import numpy as np
from sklearn.metrics import classification_report
import json

# 讀取json檔案
file_name = "epoch=98-step=41184results_jigsaw-toxic-comment-classification-challenge.json"
with open(file_name, 'r') as f:
    data = json.load(f)
# 設定threshold
threshold = 0.5
# 轉換scores為預測標籤
predicted_labels = np.array(data['scores']) > threshold
# 將targets轉換為numpy array
true_labels = np.array(data['targets'])
# 計算指標
report = classification_report(true_labels, predicted_labels, target_names=[f'Class {i}' for i in range(6)])
print(report)
'''

              precision    recall  f1-score   support

     Class 0       0.90      0.95      0.92        93
     Class 1       1.00      0.92      0.96        12
     Class 2       0.92      0.90      0.91        50
     Class 3       0.67      1.00      0.80         4
     Class 4       0.90      1.00      0.95        45
     Class 5       1.00      0.88      0.93         8

   micro avg       0.90      0.94      0.92       212
   macro avg       0.90      0.94      0.91       212
weighted avg       0.91      0.94      0.92       212
 samples avg       0.08      0.09      0.08       212
'''