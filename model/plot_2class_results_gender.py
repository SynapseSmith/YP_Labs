import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
metrics = ["Accuracy", "F1 A", "F1 C"]

# 실험 결과
added_2000 = [0.991, 0.9909, 0.9909]  # 추가 데이터 2000 결과
male_avg = [0.9872, 0.9863, 0.9879]  # 남자만 학습 결과 평균
female_avg = [0.9916, 0.9922, 0.9908]  # 여자만 학습 결과 평균

# 막대 위치 설정
x = np.arange(len(metrics))
width = 0.25  # 막대 너비

# 그래프 생성
plt.figure(figsize=(14, 7))
bars1 = plt.bar(x - width, added_2000, width, label="Total 6000 Data", color="mediumpurple", alpha=0.8)
bars2 = plt.bar(x, male_avg, width, label="Male Only", color="cornflowerblue", alpha=0.8)
bars3 = plt.bar(x + width, female_avg, width, label="Female Only", color="lightpink", alpha=0.8)

# 각 막대에 값 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=20)

# 그래프 설정
plt.xlabel("Metrics", fontsize=16)
plt.ylabel("Scores", fontsize=20)
plt.xticks(x, metrics, rotation=20, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0.925, 0.995)
plt.legend(loc="lower right", fontsize=12)
# plt.title("Comparison of Added 2000 Data, Male, and Female Results", fontsize=18)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# 그래프 출력
plt.show()