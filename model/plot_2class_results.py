import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
metrics = ["Accuracy", "Precision A", "Precision C", "Recall A", "Recall C", "F1 A", "F1 C"]
baseline = [0.923, 0.912, 0.937, 0.939, 0.908, 0.925, 0.922]
refined_c = [0.949, 0.9438, 0.9544, 0.9550, 0.9430, 0.9493, 0.9486]
added_500 = [0.966, 0.9612, 0.9714, 0.9713, 0.9606, 0.9661, 0.9659]
added_1000 = [0.989, 0.9856, 0.9925, 0.9925, 0.9865, 0.9893, 0.9891]
added_2000 = [0.991, 0.9907, 0.9913, 0.9916, 0.9907, 0.9909, 0.9909]

# 막대의 위치 설정
x = np.arange(len(metrics))
width = 0.18  # 막대 간격 조정

# 그래프 생성
plt.figure(figsize=(16, 8))
bars1 = plt.bar(x - 2 * width, baseline, width, label='Baseline (Original)', color='skyblue', alpha=0.8)
bars2 = plt.bar(x - width, refined_c, width, label='Refined C', color='lightgreen', alpha=0.8)
bars3 = plt.bar(x, added_500, width, label='Added 1000 Data', color='salmon', alpha=0.8)
bars4 = plt.bar(x + width, added_1000, width, label='Added 2000 Data', color='gold', alpha=0.8)
bars5 = plt.bar(x + 2 * width, added_2000, width, label='Added 4000 Data', color='mediumpurple', alpha=0.8)

# 각 막대에 수치 표시
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=16)

# 그래프 설정
plt.xlabel("Metrics", fontsize=18)
plt.ylabel("Scores", fontsize=18)
plt.xticks(x, metrics, rotation=20, fontsize=20)
plt.yticks(fontsize=18)
plt.ylim(0.88, 1.001)
plt.legend(loc="lower right", fontsize=15)
plt.tight_layout()

# 그래프 출력
plt.show()
