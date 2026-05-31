import numpy as np
import matplotlib.pyplot as plt

# 数据
groups = ["Truck", "People", "Car", "mAP"]
methods = ["Ours", "ResShift", "SinSR", "Bi-DiffSR"]

values = {
    "Truck":  [0.339, 0.321, 0.312, 0.298],
    "People": [0.368, 0.341, 0.361, 0.355],
    "Car":    [0.487, 0.422, 0.479, 0.469],
    "mAP":    [0.331, 0.314, 0.325, 0.319],
}

colors = {
    "Ours": "#f6b6b6",
    "ResShift": "#aeb0c8",
    "SinSR": "#f4d7b5",
    "Bi-DiffSR": "#a8d6dd"
}

# 展平数据
all_vals = []
all_methods = []
all_groups = []

for g in groups:
    for m, v in zip(methods, values[g]):
        all_groups.append(g)
        all_methods.append(m)
        all_vals.append(v)

N = len(all_vals)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
width = 2*np.pi / N

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')

# 柱子
bars = ax.bar(
    angles,
    all_vals,
    width=width,
    bottom=0,
    color=[colors[m] for m in all_methods],
    edgecolor='none',
    alpha=0.9
)

# 去掉默认角度标签
ax.set_xticks([])
ax.set_yticklabels([])
ax.grid(alpha=0.3)

# 方法名和数值标签
for angle, val, method in zip(angles, all_vals, all_methods):
    rotation = np.degrees(angle)
    ha = 'left'
    if 90 < rotation < 270:
        rotation += 180
        ha = 'right'

    ax.text(
        angle,
        val + 0.015,
        f"{method}\n{val:.3f}",
        rotation=rotation,
        rotation_mode='anchor',
        ha=ha,
        va='center',
        fontsize=10
    )

# 大类标签放在每组中心
for i, g in enumerate(groups):
    start = i * len(methods)
    end = start + len(methods)
    group_angle = np.mean(angles[start:end])

    rotation = np.degrees(group_angle)
    ha = 'center'
    if 90 < rotation < 270:
        rotation += 180

    ax.text(
        group_angle,
        max(all_vals) + 0.08,
        g,
        fontsize=22,
        fontweight='bold',
        rotation=rotation,
        rotation_mode='anchor',
        ha=ha,
        va='center'
    )

plt.title("Detection Performance", fontsize=20, fontweight='bold', pad=30)
plt.show()