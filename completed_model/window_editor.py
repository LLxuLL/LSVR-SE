import copy
import time
import traceback
import open3d as o3d
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass
from enum import Enum
import trimesh


class EditOperation(Enum):
    """编辑操作枚举"""
    ADD = "add"
    ROTATE = "rotation_angle"
    FIXED_EDGES = "fixed_edges"
    OPEN = "open"
    CLOSE = "close"
    TRANSLATE = "translate"
    SCALE = "scale"
    SLIDE_OPEN = "slide_open"

@dataclass
class WindowEditParams:
    def __init__(self, operation=None, rotation_angle=0.0, fixed_edges=None,
                 open_direction="outward", pivot_edge="left",
                 translation_vector=(0.0, 0.0, 0.0), scale_factor=1.0):
        self.operation = operation
        self.rotation_angle = rotation_angle
        self.fixed_edges = fixed_edges
        self.open_direction = open_direction
        self.pivot_edge = pivot_edge
        self.translation_vector = translation_vector
        self.scale_factor = scale_factor
        self.add_direction = "right"  # ✅ 默认方向


class SemanticWindowEditor:
    """语义窗户编辑器"""

    def __init__(self):
        self.operation_keywords = {
            EditOperation.ADD: ["添加", "增加", "新建", "create", "add", "new"],
            EditOperation.ROTATE: ["旋转", "转动", "rotate", "rotation", "turn"],
            EditOperation.FIXED_EDGES: ["固定", "锁定", "fixed", "lock", "anchor"],
            EditOperation.OPEN: ["打开", "开启", "open", "open up"],
            EditOperation.CLOSE: ["关闭", "合上", "close", "shut"],
            EditOperation.TRANSLATE: ["移动", "平移", "translate", "move", "shift"],
            EditOperation.SCALE: ["缩放", "调整大小", "scale", "resize"]
        }

        self.angle_patterns = [
            r'(\d+)\s*度',  # 匹配中文度数
            r'(\d+)\s*°',  # 匹配度数符号
            r'(\d+)\s*degree',  # 匹配英文度数
            r'rotate\s+(\d+)',  # 匹配rotate关键字
        ]

        self.edge_patterns = {
            "left": ["左边", "左侧", "left", "left side"],
            "right": ["右边", "右侧", "right", "right side"],
            "top": ["上边", "顶部", "top", "upper"],
            "bottom": ["下边", "底部", "bottom", "lower"]
        }

    def add_window(self, input_mesh: o3d.geometry.TriangleMesh, direction="right") -> o3d.geometry.TriangleMesh:
        """
        添加窗户：将当前窗户整体复制一份，并沿指定方向平移拼接
        direction: "right"（默认，向右拼接），"left"、"up"、"down"、"front"、"back"
        """
        import copy
        # 复制窗户几何体
        new_window = copy.deepcopy(input_mesh)

        # 获取尺寸信息
        min_bound = np.min(np.asarray(input_mesh.vertices), axis=0)
        max_bound = np.max(np.asarray(input_mesh.vertices), axis=0)
        size = max_bound - min_bound

        # 计算平移向量
        if direction == "right":
            translation = np.array([size[0], 0.0, 0.0])
        elif direction == "left":
            translation = np.array([-size[0], 0.0, 0.0])
        elif direction == "up":
            translation = np.array([0.0, size[1], 0.0])
        elif direction == "down":
            translation = np.array([0.0, -size[1], 0.0])
        elif direction in ("front", "前"):
            translation = np.array([0.0, 0.0, size[2]])
        elif direction in ("back", "后"):
            translation = np.array([0.0, 0.0, -size[2]])
        else:
            raise ValueError(f"未知方向: {direction}")

        # 平移复制的窗户
        new_window.translate(translation)

        # 合并原窗户和新窗户
        combined = input_mesh + new_window

        print(f"添加窗户成功，方向={direction}，偏移={translation}")
        return combined

    def extract_window_sash(self, window_mesh: o3d.geometry.TriangleMesh,
                            window_info: Dict) -> o3d.geometry.TriangleMesh:
        """从窗户模型中提取窗扇（中心玻璃部分）保留原始细节"""
        min_bounds, max_bounds = window_info["bounds"]
        width, height, depth = window_info["size"]

        # 计算窗扇边界框（保留原始细节）
        frame_thickness = min(width, height, depth) * 0.05  # 边框厚度
        sash_min_bound = np.array([
            min_bounds[0] + frame_thickness,
            min_bounds[1] + frame_thickness,
            min_bounds[2] - 0.001  # 稍微突出
        ])
        sash_max_bound = np.array([
            max_bounds[0] - frame_thickness,
            max_bounds[1] - frame_thickness,
            max_bounds[2] + depth * 0.2  # 稍微突出
        ])

        # 提取窗扇顶点
        vertices = np.asarray(window_mesh.vertices)
        triangles = np.asarray(window_mesh.triangles)

        # 找出在窗扇区域内的顶点
        in_sash_mask = np.all(
            (vertices >= sash_min_bound) & (vertices <= sash_max_bound),
            axis=1
        )

        # 创建新网格
        sash = o3d.geometry.TriangleMesh()

        if np.any(in_sash_mask):
            # 创建顶点索引映射
            sash_vertex_indices = np.where(in_sash_mask)[0]
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sash_vertex_indices)}

            # 提取窗扇三角形
            sash_triangles = []
            for tri in triangles:
                if in_sash_mask[tri[0]] and in_sash_mask[tri[1]] and in_sash_mask[tri[2]]:
                    new_tri = [vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]]
                    sash_triangles.append(new_tri)

            if sash_triangles:
                # 创建窗扇网格
                sash.vertices = o3d.utility.Vector3dVector(vertices[sash_vertex_indices])
                sash.triangles = o3d.utility.Vector3iVector(np.array(sash_triangles))
                print(f"成功提取窗扇，顶点数: {len(sash.vertices)}, 三角形数: {len(sash.triangles)}")
                return sash

        print("无法提取窗扇，使用简单长方体替代")
        # 回退方案：创建简单窗扇
        sash_width = width - 2 * frame_thickness
        sash_height = height - 2 * frame_thickness
        sash_depth = depth * 1.1
        sash = o3d.geometry.TriangleMesh.create_box(sash_width, sash_height, sash_depth)
        sash.translate([
            min_bounds[0] + frame_thickness,
            min_bounds[1] + frame_thickness,
            min_bounds[2] - sash_depth * 0.05
        ])
        sash.paint_uniform_color([0.0, 0.5, 1.0])
        return sash

    def create_window_frame(self, window_mesh, sash):
        sash_verts = np.asarray(sash.vertices)
        sash_min = sash_verts.min(axis=0) - 1e-6
        sash_max = sash_verts.max(axis=0) + 1e-6

        vertices = np.asarray(window_mesh.vertices)
        triangles = np.asarray(window_mesh.triangles)

        in_sash_mask = np.all(
            (vertices >= sash_min) & (vertices <= sash_max),
            axis=1
        )

        keep_faces = [face for face in triangles if
                      not (in_sash_mask[face[0]] and in_sash_mask[face[1]] and in_sash_mask[face[2]])]

        frame = o3d.geometry.TriangleMesh()
        frame.vertices = o3d.utility.Vector3dVector(vertices)
        frame.triangles = o3d.utility.Vector3iVector(np.array(keep_faces))
        frame.compute_vertex_normals()
        return frame

    def split_window_sash(self, sash: o3d.geometry.TriangleMesh, half="left"):
        """
        将窗扇按正面左右方向分成两半
        half: 'left' 取左半边, 'right' 取右半边
        """
        verts = np.asarray(sash.vertices)
        tris = np.asarray(sash.triangles)

        # 找窗扇的 AABB 中心（正面朝向已知的情况下，用 X 作为左右分割）
        center_x = (verts[:, 0].min() + verts[:, 0].max()) / 2.0

        if half == "left":
            mask = verts[:, 0] <= center_x
        elif half == "right":
            mask = verts[:, 0] > center_x
        else:
            raise ValueError("half参数必须是 'left' 或 'right'")

        # 分割固定半扇（三角形全在 mask 内的才保留）
        tris_half = [tri for tri in tris if np.all(mask[tri])]

        # 半扇顶点映射（避免保留另一半多余顶点）
        unique_half_verts_idx = np.unique(np.array(tris_half).flatten())
        half_verts = verts[unique_half_verts_idx]

        # 重构索引（因为 vertices 被裁剪了）
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_half_verts_idx)}
        remapped_tris = np.array([[index_map[idx] for idx in tri] for tri in tris_half], dtype=np.int32)

        half_mesh = o3d.geometry.TriangleMesh()
        half_mesh.vertices = o3d.utility.Vector3dVector(half_verts)
        half_mesh.triangles = o3d.utility.Vector3iVector(remapped_tris)
        half_mesh.compute_vertex_normals()

        return half_mesh

    def slide_open_window(self, input_mesh: o3d.geometry.TriangleMesh,
                          percent_open: float) -> o3d.geometry.TriangleMesh:
        """
        平移开窗：
        截取左半扇，沿正面左右方向平移到另一半位置。
        """
        window_info = self.identify_window_region(input_mesh)
        sash = self.extract_window_sash(input_mesh, window_info)
        frame = self.create_window_frame(input_mesh, sash)

        # 默认截取左半扇
        fixed_half = self.split_window_sash(sash, half="right")
        move_half = self.split_window_sash(sash, half="left")

        # 计算移动方向（窗户正面为最大面积平面）
        min_bounds, max_bounds = window_info["bounds"]
        size_x, size_y, size_z = window_info["size"]

        # 判定正面平面
        if size_x * size_y >= size_x * size_z and size_x * size_y >= size_y * size_z:
            # 正面是XY平面 → 左右是X方向
            move_dir = np.array([1.0, 0.0, 0.0])
            move_len = size_x / 2
        elif size_x * size_z >= size_y * size_z:
            # 正面是XZ平面 → 左右是X方向
            move_dir = np.array([1.0, 0.0, 0.0])
            move_len = size_x / 2
        else:
            # 正面是YZ平面 → 左右是Y方向
            move_dir = np.array([0.0, 1.0, 0.0])
            move_len = size_y / 2

        # 百分比转为平移长度（80% → 移动40%宽度）
        move_distance = (percent_open / 100.0) * move_len
        move_half.translate(move_dir * move_distance)

        result_mesh = frame + fixed_half + move_half
        return result_mesh

    def parse_natural_language(self, text: str) -> WindowEditParams:
        text_lower = text.lower()
        operations = []
        rotation_angle = 0.0
        fixed_edges = []
        percent_open_val = None
        add_direction_val = "right"  # 默认添加方向

        # ✅ 平移开窗模式匹配
        slide_patterns = [
            r"平移打开(\d+)%的窗户",
            r"平移打开(\d+)%窗户",
            r"平移完全打开窗户",
            r"完全平移打开窗户"
        ]
        for pat in slide_patterns:
            m = re.search(pat, text)
            if m:
                operations.append(EditOperation.SLIDE_OPEN)
                if "完全" in text:
                    percent_open_val = 100.0
                else:
                    percent_open_val = float(m.group(1))
                break

        # 匹配其余操作
        for op, keywords in self.operation_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if op not in operations:
                        operations.append(op)
                    break

        # 检测旋转角度
        for pattern in self.angle_patterns:
            match = re.search(pattern, text_lower)
            if match:
                rotation_angle = float(match.group(1))
                break

        # 检测固定边缘
        for edge, keywords in self.edge_patterns.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    fixed_edges.append(edge)
                    break

        # 检测开窗方向
        open_direction = "outward"
        if any(word in text_lower for word in ["向内", "inward", "inside"]):
            open_direction = "inward"

        # 检测旋转枢轴
        pivot_edge = "left"
        for edge in ["left", "right", "top", "bottom"]:
            if any(keyword in text_lower for keyword in self.edge_patterns[edge]):
                pivot_edge = edge
                break

        # ✅ 解析添加窗户方向
        if any(word in text for word in ["添加窗户", "加窗"]):
            if "左" in text or "left" in text:
                add_direction_val = "left"
            elif "右" in text or "right" in text:
                add_direction_val = "right"
            elif "上" in text or "up" in text:
                add_direction_val = "up"
            elif "下" in text or "down" in text:
                add_direction_val = "down"
            elif "前" in text or "front" in text:
                add_direction_val = "front"
            elif "后" in text or "back" in text:
                add_direction_val = "back"

        # ✅ 返回参数对象
        params = WindowEditParams(
            operation=operations,
            rotation_angle=rotation_angle,
            fixed_edges=fixed_edges if fixed_edges else None,
            open_direction=open_direction,
            pivot_edge=pivot_edge
        )

        # 额外赋值平移开窗百分比 & 添加方向
        if EditOperation.SLIDE_OPEN in operations:
            params.percent_open = percent_open_val if percent_open_val is not None else 100.0
        params.add_direction = add_direction_val

        return params

    def load_edit_config(self, config_path: str) -> Dict:
        """从配置文件加载编辑参数"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_edit_config(self, config: Dict, output_path: str):
        """保存编辑配置到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def identify_window_region(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """识别窗户区域"""
        vertices = np.asarray(mesh.vertices)

        # 计算边界框
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        center = (min_bounds + max_bounds) / 2

        # 假设窗户在垂直面上（Y轴为高度）
        window_height = max_bounds[1] - min_bounds[1]
        window_width = max_bounds[0] - min_bounds[0]
        window_depth = max_bounds[2] - min_bounds[2]

        # 窗户参数
        shortest_edge = min(window_width, window_height, window_depth)
        if shortest_edge == window_height:
            # Y轴是最短边，窗户在XZ平面
            window_size = (window_width, window_depth, shortest_edge)
            window_center = center
        elif shortest_edge == window_width:
            # X轴是最短边，窗户在YZ平面
            window_size = (shortest_edge, window_height, window_depth)
            window_center = center
        else:
            # Z轴是最短边，窗户在XY平面
            window_size = (window_width, window_height, shortest_edge)
            window_center = center

        return {
            "center": window_center,
            "size": window_size,  # 使用元组形式返回尺寸
            "bounds": (min_bounds, max_bounds)
        }

    def create_window_geometry(self, window_info: Dict) -> o3d.geometry.TriangleMesh:
        """创建窗户几何体"""
        center = window_info["center"]
        width, height, depth = window_info["size"]  # 解包元组

        # 创建窗户框
        window = o3d.geometry.TriangleMesh.create_box(width, height, depth)

        # 移动到正确位置
        window.translate([
            center[0] - width / 2,
            center[1] - height / 2,
            center[2] - depth / 2
        ])

        return window

    def apply_rotation(self, sash: o3d.geometry.TriangleMesh,
                       params: WindowEditParams, window_info: Dict) -> o3d.geometry.TriangleMesh:
        """
        应用旋转操作 - 围绕边缘旋转（基于窗户主要平面）
        :param sash: 窗扇模型
        :param params: 编辑参数
        :param window_info: 窗户信息
        :return: 旋转后的窗扇
        """
        min_bounds, max_bounds = window_info["bounds"]
        width, height, depth = window_info["size"]

        # 获取窗扇边界框
        sash_bbox = sash.get_axis_aligned_bounding_box()
        sash_min = sash_bbox.get_min_bound()
        sash_max = sash_bbox.get_max_bound()
        sash_center = sash_bbox.get_center()

        # 确定窗户的主要平面（最大尺寸的面）
        if width > height and width > depth:
            main_plane = "xy"
        elif height > width and height > depth:
            main_plane = "xz"
        else:
            main_plane = "yz"

        # 根据窗户主要平面和枢轴边缘确定旋转轴
        if main_plane == "xy":
            # XY平面（窗户正面）
            if params.pivot_edge == "left":
                pivot = np.array([sash_min[0], sash_center[1], sash_center[2]])
                axis = np.array([0, 1, 0])  # Y轴
            elif params.pivot_edge == "right":
                pivot = np.array([sash_max[0], sash_center[1], sash_center[2]])
                axis = np.array([0, 1, 0])  # Y轴
            elif params.pivot_edge == "top":
                pivot = np.array([sash_center[0], sash_max[1], sash_center[2]])
                axis = np.array([1, 0, 0])  # X轴
            elif params.pivot_edge == "bottom":
                pivot = np.array([sash_center[0], sash_min[1], sash_center[2]])
                axis = np.array([1, 0, 0])  # X轴
            else:
                pivot = np.array([sash_min[0], sash_center[1], sash_center[2]])
                axis = np.array([0, 1, 0])

        elif main_plane == "xz":
            # XZ平面（窗户正面）
            if params.pivot_edge == "left":
                pivot = np.array([sash_min[0], sash_center[1], sash_center[2]])
                axis = np.array([0, 0, 1])  # Z轴
            elif params.pivot_edge == "right":
                pivot = np.array([sash_max[0], sash_center[1], sash_center[2]])
                axis = np.array([0, 0, 1])  # Z轴
            elif params.pivot_edge == "top":
                pivot = np.array([sash_center[0], sash_center[1], sash_max[2]])
                axis = np.array([1, 0, 0])  # X轴
            elif params.pivot_edge == "bottom":
                pivot = np.array([sash_center[0], sash_center[1], sash_min[2]])
                axis = np.array([1, 0, 0])  # X轴
            else:
                pivot = np.array([sash_min[0], sash_center[1], sash_center[2]])
                axis = np.array([0, 0, 1])

        else:  # main_plane == "yz"
            # YZ平面（窗户正面）
            if params.pivot_edge == "left":
                pivot = np.array([sash_center[0], sash_min[1], sash_center[2]])
                axis = np.array([0, 0, 1])  # Z轴
            elif params.pivot_edge == "right":
                pivot = np.array([sash_center[0], sash_max[1], sash_center[2]])
                axis = np.array([0, 0, 1])  # Z轴
            elif params.pivot_edge == "top":
                pivot = np.array([sash_center[0], sash_center[1], sash_max[2]])
                axis = np.array([0, 1, 0])  # Y轴
            elif params.pivot_edge == "bottom":
                pivot = np.array([sash_center[0], sash_center[1], sash_min[2]])
                axis = np.array([0, 1, 0])  # Y轴
            else:
                pivot = np.array([sash_center[0], sash_min[1], sash_center[2]])
                axis = np.array([0, 0, 1])

        # 根据开窗方向调整旋转角度符号
        angle = params.rotation_angle
        if params.open_direction == "inward":
            angle = -angle

        # 应用旋转
        angle_rad = np.radians(angle)
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)

        # 围绕枢轴点旋转
        vertices = np.asarray(sash.vertices)
        translated = vertices - pivot
        rotated = translated @ rotation.T
        new_vertices = rotated + pivot

        sash.vertices = o3d.utility.Vector3dVector(new_vertices)
        sash.compute_vertex_normals()  # 重新计算法线
        return sash

    def apply_translation(self, mesh: o3d.geometry.TriangleMesh,
                          params: WindowEditParams) -> o3d.geometry.TriangleMesh:
        """应用平移操作"""
        vertices = np.asarray(mesh.vertices)
        new_vertices = vertices + np.array(params.translation_vector)
        mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        return mesh

    def apply_scaling(self, mesh: o3d.geometry.TriangleMesh,
                      params: WindowEditParams, center: np.ndarray) -> o3d.geometry.TriangleMesh:
        """应用缩放操作"""
        vertices = np.asarray(mesh.vertices)
        translated = vertices - center
        scaled = translated * params.scale_factor
        new_vertices = scaled + center
        mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        return mesh

    def process_from_text(self, input_mesh_path: str, text_command: str,
                          output_path: str) -> bool:
        """从文本命令处理3D模型"""
        try:
            # 加载输入网格
            mesh = o3d.io.read_triangle_mesh(input_mesh_path)
            if not mesh.has_vertices():
                print("无法加载网格或网格为空")
                return False

            # 解析自然语言
            edit_params = self.parse_natural_language(text_command)

            # 执行编辑
            edited_mesh = self.edit_window(mesh, edit_params)

            # 确保编辑后的网格有效
            if edited_mesh is None or len(edited_mesh.vertices) == 0:
                print("编辑后的网格无效，保存原始网格")
                edited_mesh = mesh

            # 保存结果
            o3d.io.write_triangle_mesh(output_path, edited_mesh)
            print(f"编辑完成，结果保存至: {output_path}")

            # 新增：可视化结果
            self.visualize_edit_result(edited_mesh, text_command)

            return True

        except Exception as e:
            print(f"处理失败: {str(e)}")
            traceback.print_exc()
            return False

    def visualize_edit_result(self, mesh, command):
        """可视化编辑结果"""
        try:
            import pyvista as pv
            from stpyvista import stpyvista

            # 创建PyVista网格
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            # 确保面是三角形
            if faces.size > 0 and faces.shape[1] == 3:
                # 添加连接信息
                faces = np.insert(faces, 0, 3, axis=1)

                # 创建PyVista网格
                pv_mesh = pv.PolyData(vertices, faces)

                # 创建绘图器
                plotter = pv.Plotter(window_size=[400, 400])
                plotter.add_mesh(pv_mesh, color='tan', show_edges=True)
                plotter.add_title(f"编辑结果: {command}", font_size=10)
                plotter.view_isometric()

                # 保存到临时文件
                temp_file = "./temp_edit_visualization.png"
                plotter.screenshot(temp_file)
                plotter.close()

                # 在Streamlit中显示
                stpyvista(plotter, key=f"edit_{time.time()}")

                return True
            else:
                print("无法创建有效的网格面数据")
                return False
        except Exception as e:
            print(f"可视化失败: {str(e)}")
            return False

    def _get_operation_from_string(self, op_str: str) -> EditOperation:
        """将字符串映射到 EditOperation 枚举"""
        mapping = {
            "add": EditOperation.ADD,
            "rotation_angle": EditOperation.ROTATE,
            "fixed_edges": EditOperation.FIXED_EDGES,
            "open": EditOperation.OPEN,
            "close": EditOperation.CLOSE,
            "translate": EditOperation.TRANSLATE,
            "scale": EditOperation.SCALE,
            "rotate": EditOperation.ROTATE,  # 添加 "rotate" 的映射
        }
        return mapping.get(op_str.lower(), EditOperation.ADD)

    def process_from_config(self, input_mesh_path: str, config_path: str,
                            output_path: str) -> bool:
        """从配置文件处理3D模型"""
        try:
            config = self.load_edit_config(config_path)

            # 构建编辑参数
            params = WindowEditParams(
                operation=[self._get_operation_from_string(op) for op in config.get("operations", [])],
                rotation_angle=config.get("rotation_angle", 0.0),
                fixed_edges=config.get("fixed_edges", None),
                open_direction=config.get("open_direction", "outward"),
                pivot_edge=config.get("pivot_edge", "left"),
                translation_vector=tuple(config.get("translation", [0.0, 0.0, 0.0])),
                scale_factor=config.get("scale", 1.0)
            )

            # 加载并编辑网格
            mesh = o3d.io.read_triangle_mesh(input_mesh_path)
            if not mesh.has_vertices():
                print(f"无法加载网格或网格为空: {input_mesh_path}")
                return False

            edited_mesh = self.edit_window(mesh, params)

            # 保存结果
            o3d.io.write_triangle_mesh(output_path, edited_mesh)
            print(f"配置编辑完成，结果保存至: {output_path}")

            return True

        except Exception as e:
            print(f"配置处理失败: {str(e)}")
            return False

    def edit_window(self, input_mesh: o3d.geometry.TriangleMesh,
                    edit_params: WindowEditParams) -> o3d.geometry.TriangleMesh:
        if not edit_params.operation:
            print("未检测到编辑操作，保持原样")
            return input_mesh

        print(f"检测到的操作: {[op.value for op in edit_params.operation]}")
        window_info = self.identify_window_region(input_mesh)
        edited_mesh = copy.deepcopy(input_mesh)

        #1. 平移开窗优先
        if EditOperation.SLIDE_OPEN in edit_params.operation:
            percent = getattr(edit_params, "percent_open", 100.0)
            print(f"执行平移开窗: 打开{percent}%")
            try:
                return self.slide_open_window(input_mesh, percent)
            except Exception as e:
                print(f"平移开窗失败: {str(e)}")
                return input_mesh

        # 开窗操作
        if EditOperation.OPEN in edit_params.operation:
            print(
                f"执行开窗操作: 方向={edit_params.open_direction}, 枢轴={edit_params.pivot_edge}, 角度={edit_params.rotation_angle}°")
            try:
                # 1. 提取窗扇
                sash = self.extract_window_sash(input_mesh, window_info)

                # 2. 从原窗户挖空窗扇 => 得到空心窗框（用修改后的 create_window_frame）
                frame = self.create_window_frame(input_mesh, sash)

                # 3. 旋转窗扇
                rotated_sash = self.apply_rotation(sash, edit_params, window_info)

                # 4. 组合框和旋转后的窗扇
                edited_mesh = frame + rotated_sash

            except Exception as e:
                print(f"开窗操作失败: {str(e)}")
                edited_mesh = input_mesh

        # 处理关窗操作
        if EditOperation.CLOSE in edit_params.operation:
            print(f"执行关窗操作")

            try:
                # 步骤1: 提取窗扇
                sash = self.extract_window_sash(edited_mesh, window_info)

                # 步骤2: 创建窗框
                frame = self.create_window_frame(edited_mesh, sash)

                # 步骤3: 设置旋转角度为0（关闭位置）
                edit_params.rotation_angle = 0.0

                # 步骤4: 将窗扇旋转回原位
                closed_sash = self.apply_rotation(sash, edit_params, window_info)

                # 步骤5: 组合窗框和关闭的窗扇
                edited_mesh = frame + closed_sash

            except Exception as e:
                print(f"关窗操作失败: {str(e)}")
                edited_mesh = input_mesh

        # 处理普通旋转操作
        if EditOperation.ROTATE in edit_params.operation:
            print(f"执行旋转操作: 角度={edit_params.rotation_angle}度, 枢轴={edit_params.pivot_edge}")

            try:
                # 步骤1: 提取窗扇
                sash = self.extract_window_sash(edited_mesh, window_info)

                # 步骤2: 创建窗框
                frame = self.create_window_frame(edited_mesh, sash)

                # 步骤3: 旋转窗扇
                rotated_sash = self.apply_rotation(sash, edit_params, window_info)

                # 步骤4: 组合窗框和旋转后的窗扇
                edited_mesh = frame + rotated_sash

            except Exception as e:
                print(f"旋转操作失败: {str(e)}")
                edited_mesh = input_mesh

        # 处理其他操作...
        if EditOperation.ADD in edit_params.operation:
            direction = getattr(edit_params, "add_direction", "right")
            print(f"执行添加窗户操作，方向={direction}")
            try:
                edited_mesh = self.add_window(input_mesh, direction=direction)
            except Exception as e:
                print(f"添加窗户失败: {str(e)}")
                edited_mesh = input_mesh

        if EditOperation.TRANSLATE in edit_params.operation:
            if edited_mesh is not None:
                edited_mesh = self.apply_translation(edited_mesh, edit_params)

        if EditOperation.SCALE in edit_params.operation:
            if edited_mesh is not None:
                center = window_info["center"]
                edited_mesh = self.apply_scaling(edited_mesh, edit_params, center)

        # 确保返回有效的网格
        if edited_mesh is None or len(edited_mesh.vertices) == 0:
            print("编辑后的网格无效，返回原始网格")
            return input_mesh

        return edited_mesh

def test_semantic_editor():
    """测试语义编辑器 - 优化版本"""
    editor = SemanticWindowEditor()

    # 测试自然语言解析
    test_commands = [
        "请把窗户向外旋转75度打开",
        "将窗户向左旋转45度",
        "添加一个新窗户并固定顶部边缘",
        "关闭当前窗户",
        "把窗户向右移动10个单位",
        "平移打开80%的窗户"
    ]

    for cmd in test_commands:
        params = editor.parse_natural_language(cmd)
        print(f"\n命令: {cmd}")
        print(f"解析结果: {params}")

    # 加载指定路径的窗户模型
    window_path = "./output/edited_input/window.ply"
    print(f"加载窗户模型: {window_path}")
    test_window = o3d.io.read_triangle_mesh(window_path)

    if not test_window.has_vertices():
        print(f"错误: 无法加载窗户模型: {window_path}")
        print("请检查文件路径和文件格式")
        return False

    # 可视化原始窗户
    o3d.visualization.draw_geometries([test_window], window_name="原始窗户")

    # 测试编辑操作
    output_dir = "./output/edited_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 保存原始模型
    original_path = os.path.join(output_dir, "original_window.ply")
    o3d.io.write_triangle_mesh(original_path, test_window)
    print(f"原始模型已保存至: {original_path}")

    # 测试开窗操作
    print("\n测试开窗操作...")
    opened_path = os.path.join(output_dir, "edited_window.ply")
    success = editor.process_from_text(
        original_path,
        "在前边添加窗户",
        opened_path
    )

    if success:
        try:
            opened_mesh = o3d.io.read_triangle_mesh(opened_path)
            o3d.visualization.draw_geometries([opened_mesh], window_name="开窗结果")
        except:
            print("无法加载或可视化开窗结果")

    # 测试关窗操作
    print("\n测试关窗操作...")
    closed_path = os.path.join(output_dir, "closed_window.ply")
    success = editor.process_from_text(
        opened_path if os.path.exists(opened_path) else original_path,
        "关闭当前窗户",
        closed_path
    )

    if success:
        try:
            closed_mesh = o3d.io.read_triangle_mesh(closed_path)
            o3d.visualization.draw_geometries([closed_mesh], window_name="关窗结果")
        except:
            print("无法加载或可视化关窗结果")

    # 测试配置文件
    config = {
        "operations": ["open"],
        "rotation_angle": 90,
        "open_direction": "outward",
        "pivot_edge": "left"
    }

    config_path = os.path.join(output_dir, "edit_config.json")
    editor.save_edit_config(config, config_path)

    config_output_path = os.path.join(output_dir, "config_edited_window.ply")
    success = editor.process_from_config(
        original_path,
        config_path,
        config_output_path
    )

    if success:
        try:
            config_mesh = o3d.io.read_triangle_mesh(config_output_path)
            o3d.visualization.draw_geometries([config_mesh], window_name="配置编辑结果")
        except:
            print("无法加载或可视化配置编辑结果")

    return success


if __name__ == "__main__":
    # 运行测试
    test_semantic_editor()