# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 下午4:08
# @Author  : sjh
# @Site    : 
# @File    : config.py
# @Comment :
import numpy as np
import cv2
import json
import open3d as o3d
from matplotlib import pyplot as plt


class CameraIntrinsics:
    def read_calib(self, calib_path):
        # 读取标定文件
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)

        left_k = np.array(calib_data['k_left'])
        right_k = np.array(calib_data['k_right'])
        left_distortion = np.array(calib_data['dist_left'])
        right_distortion = np.array(calib_data['dist_right'])
        # left_Q = np.array(calib_data['left_camera_intrinsics']['Q_left'])
        # right_Q = np.array(calib_data['right_camera_intrinsics']['Q_right'])
        rotation_matrix = np.array(calib_data['R'])
        translation_vector = np.array(calib_data['T'])
        Q_matrix = np.array(calib_data['Q'])
        # inverse_extrinsic = np.array(calib_data['extrinsic']['inverse_extrinsic_matrix'])
        # R = inverse_extrinsic[:3, :3]
        # T = inverse_extrinsic[:3, 3]
        # T = T.reshape(3, 1)
        # T = np.array(calib_data['extrinsic']['t'])

        return left_k, right_k, left_distortion, right_distortion, rotation_matrix, translation_vector, Q_matrix

    def getIntrinsics1920_1080(self):
        height = 1080
        width = 1920
        p = [
            788.4632328, 0.0, 961.03165436, 33.82135389646828,
            0.0, 788.70885008, 527.51271498, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # 635.982 715.47975 719.8649368286133 464.5906677246094 70.90622931718826
        p = [
            635.982, 0.0, 719.8649368286133, 0,
            0.0, 715.47975, 464.5906677246094, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        baseline = 0.07090622931718826
        return height, width, p, baseline

    def getIntrinsics1280_640(self):
        height = 640
        width = 1280

        p = [
            476.987060546875, 0.0, 459.9617614746094, 33.82135389646828,
            0.0, 423.9884948730469, 275.3126220703125, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        p = [
            423.988, 0.0, 479.9099578857422, 33.82135389646828,
            0.0, 423.988, 275.31298828125, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # p = [
        #     460.3950380000503, 0.0, 620.1596069335938, 33.82135389646828,
        #     0.0, 460.3950380000503, 267.4479331970215, 0.0,
        #     0.0, 0.0, 1.0, 0.0
        # ]
        baseline = 0.07090622931718826
        return height, width, p, baseline

    def getIntrinsics640_480(self):
        height = 480
        width = 640
        [fx, fy, cx, cy, baseline] = [252.562, 252.562, 309.352, 147.295, 70.0427]

        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # 矫正后
        fx = 229.98088073730466
        fy = 229.98088073730466
        cx = 329.4670867919922
        cy = 206.48446655273438
        fx = 229.98088073730466
        fy = 229.98088073730466
        cx = 318.0451965332031
        cy = 206.48446655273438
        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        baseline = 0.0700427
        return height, width, p, baseline

    def getIntrinsics640_352(self):
        height = 352
        width = 640

        p = [
            238.4935302734375, 0.0, 233.19366455078125, 33.82135389646828,
            0.0, 229.9808807373047, 151.42193603515625, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        baseline = 0.07090622931718826
        return height, width, p, baseline

    def getIntrinsics_AI(self):
        height = 784
        width = 1056

        left_k, right_k, left_distortion, right_distortion, r, t, q = self.read_calib(r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\cali_circle155.json")
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(left_k, left_distortion, right_k, right_distortion,
                                                          (
                                                          width, height), r, t, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)
        fx, fy, cx, cy = p1[0, 0], p1[1, 1], p1[0, 2], p1[1, 2]
        baseline = abs(1 / q[3, 2]) if q[3, 2] != 0 else 0
        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        map1x, map1y = cv2.initUndistortRectifyMap(left_k, left_distortion, r1, p1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_k, right_distortion, r2, p2, (width, height), cv2.CV_32FC1)
        return height, width, p, baseline, q, map1x, map1y, map2x, map2y

    def getIntrinsics_AI155(self):
        height = 480
        width = 640

        left_k, right_k, left_distortion, right_distortion, r, t, q = self.read_calib(r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\calib-circle155.json")
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(left_k, left_distortion, right_k, right_distortion,
                                                          (width, height), r, t, alpha=0)
        fx, fy, cx, cy = p1[0, 0], p1[1, 1], p1[0, 2], p1[1, 2]
        baseline = abs(1 / q[3, 2]) if q[3, 2] != 0 else 0
        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        map1x, map1y = cv2.initUndistortRectifyMap(left_k, left_distortion, r1, p1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_k, right_distortion, r2, p2, (width, height), cv2.CV_32FC1)
        return height, width, p, baseline, q, map1x, map1y, map2x, map2y

    def getIntrinsics_AI640(self):
        height = 480
        width = 640

        left_k, right_k, left_distortion, right_distortion, r, t, q = self.read_calib(r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\calib-circle640x480.json")
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(left_k, left_distortion, right_k, right_distortion,
                                                          (width, height), r, t, alpha=0)
        fx, fy, cx, cy = p1[0, 0], p1[1, 1], p1[0, 2], p1[1, 2]
        baseline = abs(1 / q[3, 2]) if q[3, 2] != 0 else 0
        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        map1x, map1y = cv2.initUndistortRectifyMap(left_k, left_distortion, r1, p1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_k, right_distortion, r2, p2, (width, height), cv2.CV_32FC1)
        return height, width, p, baseline, q, map1x, map1y, map2x, map2y

    def getIntrinsics_uvc(self):
        height = 480
        width = 640

        left_k, right_k, left_distortion, right_distortion, r, t, q = self.read_calib(r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\cali_circleuvc.json")
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(left_k, left_distortion, right_k, right_distortion,
                                                          (width, height), r, t, alpha=0)
        fx, fy, cx, cy = p1[0, 0], p1[1, 1], p1[0, 2], p1[1, 2]
        baseline = abs(1 / q[3, 2]) if q[3, 2] != 0 else 0
        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        baseline = baseline
        return height, width, p, baseline, q


class Stereo:
    def __init__(self, resolution=(640, 480)):
        ori_height, ori_width, p, baseline, q, self.map1x, self.map1y, self.map2x, self.map2y = CameraIntrinsics().getIntrinsics_AI155()
        self.fx, self.fy, self.cx, self.cy, self.baseline = p[0], p[5], p[2], p[6], baseline * 1000
        self.Q = q
        self.depth_cam_matrix = np.array([[self.fx, 0, self.cx],
                                          [0, self.fy, self.cy],
                                          [0, 0, 1]])
        self.focal_length = self.depth_cam_matrix[0, 0]  # 768.80165

        self.depth_map = None
        # scale_x = ori_width /resolution[0]  # 宽度缩放比例
        # scale_y = ori_height / resolution[1]  # 高度缩放比例
        scale_x = resolution[0] / ori_width  # 宽度缩放比例
        scale_y = resolution[1] / ori_height  # 高度缩放比例
        print(self.fx, self.fy, self.cx, self.cy, self.baseline)
        self.Q = self.scale_Q_matrix(self.Q, scale_x, scale_y)
        self.reset_calib(scale_x, scale_y)  # 调整内参

        print(self.fx, self.fy, self.cx, self.cy, self.baseline, 'aaaaaa')

    def reset_calib(self, scale_x, scale_y):
        self.fx, self.fy, self.cx, self.cy = self.fx * scale_x, self.fy * scale_y, self.cx * scale_x, self.cy * scale_y
        self.depth_cam_matrix = np.array([[self.fx, 0, self.cx],
                                          [0, self.fy, self.cy],
                                          [0, 0, 1]])
        print(self.depth_cam_matrix, 'bbbbbbb')

    def scale_Q_matrix(self, Q, scale_x, scale_y):
        Q_new = Q.copy()
        Q_new[0, 3] *= scale_x  # -cx
        Q_new[1, 3] *= scale_y  # -cy
        Q_new[2, 3] *= 1.0  # f 保持不变，Z=fx*B/d，其中 fx 是缩放后的

        Q_new[0, 0] *= scale_x  # X = (x - cx) * Z / fx
        Q_new[1, 1] *= scale_y  # Y = (y - cy) * Z / fy
        Q_new[2, 2] *= 1.0  # 通常是 0，没用

        Q_new[3, 2] /= scale_x  # -1 / B'，B 随 fx 比例变化，确保 Z = fx*B/d

        return Q_new

    def apply_rectification(self, left_img, right_img):

        # 1. 重映射
        rectified_left = cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR)

        return rectified_left, rectified_right

    def filter_depth(self, depth_map):
        max_depth = 10  # 单位 m
        depth_scale = 1000  # 假设深度图以 mm 存储
        depth_map = np.where(depth_map / depth_scale > max_depth, 0, depth_map)
        depth_map = np.where(depth_map / depth_scale < 0, 0, depth_map)
        # depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        return depth_map

    def save_depth(self, disp):
        # 将视差图转换为深度图，避免除零问题
        epsilon = 1e-6
        # 计算深度图（单位为米），防止除零
        depth = (self.focal_length * self.baseline) / (disp + epsilon)

        # 显示归一化后的深度图（仅用于展示，转换为 8 位）
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = np.squeeze(depth_normalized.astype(np.uint8))
        plt.imshow(depth_uint8, cmap="jet")
        plt.show()

        # 保存原始深度图（转换为毫米后保存为 16 位 PNG）
        # 这里假设 depth 单位为米，乘以 1000 转为毫米
        depth_mm = np.squeeze(depth)
        depth_16bit = depth_mm.astype(np.uint16)
        cv2.imwrite('./runs/depth_16bit.png', depth_16bit)

    def visualize_disp(self, disp, colormap=cv2.COLORMAP_MAGMA):
        norm = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
        # # 归一化到 0-255
        # depth_min = 0.3376  # np.min(depth_filtered)
        # depth_max = 20.0000  # np.max(depth_filtered)
        # depth_norm = (depth_filtered - depth_min) / (depth_max - depth_min)  # 归一化到 0-1
        # depth_vis = (depth_norm * 255).astype(np.uint8)  # 转换为 0-255 范围

        # # 伪彩色映射
        # depth_colormap = cv2.applyColorMap(depth_vis, colormap)
        return depth_colormap

    def visualize_depth(self, depth_filtered, colormap=cv2.COLORMAP_JET):
        # 归一化到 0-255
        depth_min = np.min(depth_filtered)
        depth_max = np.max(depth_filtered)
        depth_norm = (depth_filtered - depth_min) / (depth_max - depth_min)  # 归一化到 0-1
        depth_vis = (depth_norm * 255).astype(np.uint8)  # 转换为 0-255 范围

        # 伪彩色映射
        depth_colormap = cv2.applyColorMap(depth_vis, colormap)
        return depth_colormap

    def on_mouse(self, event, x, y, flags, param):
        """ 鼠标点击事件，获取像素点的 3D 坐标 """

        if event == cv2.EVENT_LBUTTONDOWN:

            point_3d = self.xy_3d(x, y, self.depth_map, self.depth_cam_matrix)
            if None in point_3d:
                print(f"点 ({x}, {y}) 的深度无效")
            else:
                print(f"点 ({x}, {y}) 的三维坐标: X={point_3d[0]:.3f}, Y={point_3d[1]:.3f}, Z={point_3d[2]:.3f} m")

    def xy_3d(self, x, y, depth_map=None, depth_cam_matrix=None, depth_scale=1000):
        """ 将图像坐标 (x, y) 转换为 3D 世界坐标 """
        fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
        cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

        z = depth_map[y, x] / depth_scale  # 获取像素点的深度值 (m)
        # print(z, self.disp[y,x],'aaaaaaa')
        if z <= 0:
            return np.array([None, None, None])  # 如果深度无效，返回空

        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy

        return np.array([X, Y, z])

    def disparity_to_depth(self, disp, focal_length, baseline):
        """ 视差图转换为深度图 """
        depth = (focal_length * baseline) / (disp + 1e-8)
        return depth

    def disparity_to_depth2(self, disp, focal_length, baseline):
        """
        使用 Q 矩阵将视差图转换为 3D 点云，并从中提取深度图
        :param disp: 视差图 (H, W)，np.float32 或 np.float64
        :return: 深度图 (H, W)，单位与 Q 一致（通常为米）
        """
        # 检查 Q 是否已加载
        if self.Q is None:
            raise ValueError("Q matrix not initialized. Make sure to run stereoRectify first.")
        if disp.dtype != np.float32:
            disp = disp.astype(np.float32)
        # 通过 Q 重建点云（X, Y, Z）
        points_3d = cv2.reprojectImageTo3D(disp, self.Q)

        # 提取 Z 轴作为深度图（左相机坐标系下的 Z，即前方向）
        depth = -points_3d[:, :, 2] * 1000

        # 可选：将负值（无效或错配区域）设为 0 或 NaN
        depth[depth <= 0] = 0  # 或 np.nan，根据下游使用情况

        return depth

    def depth2xyz(self, depth_map, depth_cam_matrix, flatten=False, depth_scale=1):
        """ 将深度图转换为点云 """
        fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
        cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

        # 生成网格
        h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]

        # 计算 3D 坐标
        z = depth_map / depth_scale  # 归一化深度（单位: m）
        x = (w - cx) * z / fx
        y = (h - cy) * z / fy
        xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)

        return xyz

    def rectify_image(self, left_img, right_img, ):
        rectifyed_left = cv2.remap(left_img, self.map_1x, self.map_1y, cv2.INTER_LINEAR)
        rectifyed_right = cv2.remap(right_img, self.map_2x, self.map_2y, cv2.INTER_LINEAR)
        return rectifyed_left, rectifyed_right

    def disp_combine(self, disp1, rectifyed_left):
        self.disp = disp1
        # if self.depth_map is None:
        self.depth_map = self.disparity_to_depth(disp1, self.fx, self.baseline)
        depth_colormap = self.visualize_disp(disp1)
        rectifyed_left = cv2.resize(rectifyed_left, (depth_colormap.shape[1], depth_colormap.shape[0]))
        # combined_image = np.hstack((rectifyed_left[:,64:], depth_colormap[:,64:]))
        combined_image = np.hstack((rectifyed_left, depth_colormap))
        cv2.imshow("Estimated disparity", combined_image)
        cv2.waitKey(1)

    def show_depth_point(self, disp1, rectifyed_left, scale=1):
        self.scale = scale

        rectifyed_left = cv2.resize(rectifyed_left, (640, 480))
        self.depth_map = self.disparity_to_depth(disp1, self.fx, self.baseline)

        max_depth = 10  # 单位 m
        depth_scale = 1000  # 假设深度图以 mm 存储
        self.depth_map = np.where(self.depth_map / depth_scale > max_depth, 0, self.depth_map)
        self.depth_map = np.where(self.depth_map / depth_scale < 0, 0, self.depth_map)
        depth_colormap = self.visualize_disp(disp1)
        rectifyed_left = cv2.resize(rectifyed_left, (depth_colormap.shape[1], depth_colormap.shape[0]))
        print(rectifyed_left.shape, depth_colormap.shape)
        combined_image = np.hstack((rectifyed_left, depth_colormap))

        while True:
            print(combined_image.shape, 'depth_colormap.shape')
            cv2.imshow("Estimated disparity", combined_image)
            cv2.setMouseCallback("Estimated disparity", self.on_mouse, 0)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            show_ply = True
            if show_ply:
                pc = self.depth2xyz(self.depth_map, self.depth_cam_matrix, flatten=True)

                # **可视化点云并允许选点**
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)

                vis = o3d.visualization.VisualizerWithEditing()
                vis.create_window(window_name="Click to Get Depth")
                vis.add_geometry(pcd)

                print("\n请在窗口中 **按住Shift + 左键** 选点，然后关闭窗口后查看选中的点。\n")

                vis.run()  # 运行可视化（允许选点）
                vis.destroy_window()

                # **获取选中的点**
                picked_points = vis.get_picked_points()
                if picked_points:
                    print("\n选中的 3D 点坐标：")
                    for i, idx in enumerate(picked_points):
                        x, y, z = pc[idx]
                        print(f"点 {i + 1}: X={x:.3f}, Y={y:.3f}, Z={z:.3f} m")
                else:
                    print("未选中任何点。")


if __name__ == '__main__':
    CameraIntrinsics = Stereo()
    # print(CameraIntrinsics.getIntrinsics1280_640())