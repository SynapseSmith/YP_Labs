import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
metrics = ["Accuracy", "Precision Normal", "Precision Null", "Precision NSFW", "Recall Normal", "Recall Null", "Recall NSFW"]

# 새 데이터
baseline = [0.9863, 0.9811, 0.9832, 0.9947, 0.9863, 0.9769, 0.9957]
refined_1000 = [0.9655, 0.9393, 0.9606, 0.9962, 0.9633, 0.9378, 0.9953]
refined_2000 = [0.9577, 0.9249, 0.9526, 0.9952, 0.9545, 0.9228, 0.9957]

# 막대의 위치 설정
x = np.arange(len(metrics))
width = 0.25

# 그래프 생성
plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - width, baseline, width, label='Data Augmentation (6,000 each)', color='slateblue', alpha=0.8)
bars2 = plt.bar(x, refined_1000, width, label='Refined Null (1,000 Null Replaced)', color='mediumseagreen', alpha=0.8)
bars3 = plt.bar(x + width, refined_2000, width, label='Refined Null (2,000 Null Replaced)', color='darkorange', alpha=0.8)

# 각 막대에 수치 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=12)

# 그래프 설정
plt.xlabel("Metrics", fontsize=18)
plt.ylabel("Scores", fontsize=18)
plt.xticks(x, metrics, rotation=20, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0.8, 1.01)
plt.legend(loc="lower right", fontsize=15)
plt.tight_layout()

# 그래프 출력
plt.show()
