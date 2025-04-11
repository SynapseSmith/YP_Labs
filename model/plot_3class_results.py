import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
metrics = ["Accuracy", "Precision Normal", "Precision Null", "Precision NSFW", "Recall Normal", "Recall Null", "Recall NSFW"]
baseline = [0.9196, 0.8797, 0.9135, 0.9681, 0.9333, 0.8577, 0.9677]
refined_nsfw = [0.9390, 0.9120, 0.9154, 0.9898, 0.9163, 0.9060, 0.9946]
refined_null = [0.9725, 0.9619, 0.9638, 0.9920, 0.9663, 0.9564, 0.9950]

# 막대의 위치 설정
x = np.arange(len(metrics))
width = 0.25

# 그래프 생성
plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - width, baseline, width, label='Baseline (Original)', color='skyblue', alpha=0.8)
bars2 = plt.bar(x, refined_nsfw, width, label='Refined NSFW Data', color='lightgreen', alpha=0.8)
bars3 = plt.bar(x + width, refined_null, width, label='Refined Null & NSFW Data', color='salmon', alpha=0.8)

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
