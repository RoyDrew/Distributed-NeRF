import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义两个姿态的旋转矩阵和平移矩阵
R1 = np.array([[0.29257137, 0.8297249, -0.475351],
               [-0.9562293, 0.2511338, -0.15019082],
               [-0.00524036, 0.4984861, 0.86688185]])

t1 = np.array([0.31210223, -0.15876806, 0.00098995])

R2 = np.array([[0.2934315, 0.829544, -0.47513652],
               [-0.9559644, 0.25177047, -0.15081005],
               [-0.00547822, 0.49846604, 0.8668919]])

t2 = np.array([0.48323432, -0.0904438, 0.00103927])

# 定义XYZ比例
scale_xyz = np.array([0.8, 36.45, 14])

# 定义比例因子
scale_factor = 0.16059734253231198

# 计算旋转矩阵之间的距离
r1 = R.from_matrix(R1)
r2 = R.from_matrix(R2)
rotation_distance = np.arccos(np.trace(np.dot(r1.as_matrix(), r2.as_matrix().T) - 1) / 2)

# 计算平移矩阵之间的距离，考虑XYZ比例和比例因子
translation_distance = np.linalg.norm((t1 - t2) * scale_xyz * scale_factor)

# 计算总体距离
pose_distance = rotation_distance + translation_distance

# print("旋转矩阵之间的距离：", rotation_distance)
# print("平移矩阵之间的距离（考虑XYZ比例和比例因子）：", translation_distance)
# print("总体姿态距离：", pose_distance)



# 定义 t1、t2 和 t3
t1 = np.array([0.31210223, -0.15876806, 0.00098995])
t2 = np.array([0.48323432, -0.0904438, 0.00103927])

alpha = 0.5  # 你可以根据需要调整 alpha

# 使用线性插值计算 t3
t3 = (1 - alpha) * t1 + alpha * t2

# 计算 t3 到 t1 和 t2 的距离
distance_t3_to_t1 = np.linalg.norm(t3 - t1)
distance_t3_to_t2 = np.linalg.norm(t3 - t2)

# 计算逆距离权重
weight_t1 = 1 / (distance_t3_to_t1 + 1e-6)  # 避免除以零
weight_t2 = 1 / (distance_t3_to_t2 + 1e-6)


# 规范化权重，使它们总和为1
total_weight = weight_t1 + weight_t2
weight_t1_normalized = weight_t1 / total_weight
weight_t2_normalized = weight_t2 / total_weight

# # 对 t1 和 t2 进行逆距离加权平均
# weighted_average_t3 = (weight_t1 * t1 + weight_t2 * t2) / (weight_t1 + weight_t2)

print("t3 到 t1 的距离:", distance_t3_to_t1)
print("t3 到 t2 的距离:", distance_t3_to_t2)
print("归一化后的权重 weight_t1_normalized:", weight_t1_normalized)
print("归一化后的权重 weight_t2_normalized:", weight_t2_normalized)
# print("逆距离加权平均的 t3:", weighted_average_t3)
print("t1:", t1)
print("t2:", t2)
print("alpha:",alpha,",t3 插值后的值:",t3)