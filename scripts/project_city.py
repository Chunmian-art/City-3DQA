import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# 读取 TXT 文件
points = []
colors = []
with open("/workspace/urbanbis/CityQA/Lihu/Area1.txt", "r") as file:
    for line in file:
        x, y, z, r, g, b, _, _, _ = map(float, line.split())
        points.append([x, y, z])
        colors.append([r / 255, g / 255, b / 255])  # 将颜色从 [0, 255] 转换到 [0, 1]

# 创建点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

# 可视化点云
# o3d.visualization.draw_geometries([point_cloud])

exit()

# 创建3D点云
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# 创建相机参数
camera_params = o3d.camera.PinholeCameraParameters()
camera_params.intrinsic.set_intrinsics(640, 480, 525.0, 525.0, 320.0, 240.0)

# 将点云投影到2D平面
projected_points, colors = o3d.camera.project_points(point_cloud.points, camera_params)

# 将投影后的点云转换为Matplotlib图像
plt.scatter(projected_points[:, 0], projected_points[:, 1], c=colors)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('3D Point Cloud Projection')
plt.show()


