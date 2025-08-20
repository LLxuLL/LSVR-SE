import copy
import json
import os
import re
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class UniversalModelEditor:
    """Universal model editor"""

    def __init__(self, config_dir: str = "./models/model_configs"):
        self.config_dir = config_dir
        self.configs: Dict[str, Dict] = {}
        self.atomic_operations = self._initialize_atomic_operations()
        self.load_all_configs()

    def _initialize_atomic_operations(self) -> Dict[str, callable]:
        """Initialize all atomic operations"""
        operations = {
            # Basic grid operations
            "copy_mesh": self._atomic_copy_mesh,
            "get_bounds": self._atomic_get_bounds,
            "translate_mesh": self._atomic_translate_mesh,
            "rotate_mesh": self._atomic_rotate_mesh,
            "rotate_mesh_around_pivot": self._atomic_rotate_mesh_around_pivot,
            "scale_mesh": self._atomic_scale_mesh,
            "combine_meshes": self._atomic_combine_meshes,

            # Mesh extraction and segmentation
            "extract_submesh_by_ratio": self._atomic_extract_submesh_by_ratio,
            "extract_inverse_submesh": self._atomic_extract_inverse_submesh,
            "split_mesh_by_axis": self._atomic_split_mesh_by_axis,
            "separate_extraction": self._atomic_separate_extraction,

            # Calculate the operation
            "calculate_translation_vector": self._atomic_calculate_translation_vector,
            "calculate_pivot_point": self._atomic_calculate_pivot_point,
            "calculate_rotation_axis": self._atomic_calculate_rotation_axis,
            "calculate_scale_center": self._atomic_calculate_scale_center,
            "calculate_slide_direction": self._atomic_calculate_slide_direction,
            "calculate_slide_distance": self._atomic_calculate_slide_distance,

            # Adjust the operation
            "adjust_angle_by_direction": self._atomic_adjust_angle_by_direction,
            "reset_mesh_position": self._atomic_reset_mesh_position,

            # New Actions
            "align_sash_to_frame": self._align_sash_to_frame,
        }
        return operations

    def load_all_configs(self):
        """Load all model profiles"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            return

        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.config_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    self.configs.update(config)
                    print(f"Loaded config from {filename}")
                except Exception as e:
                    print(f"Error loading config {filename}: {str(e)}")

    def _resolve_parameter_value(self, param_value: Any, context: Dict[str, Any]) -> Any:
        """Parsing parameter values and supporting variable references and expressions"""
        if isinstance(param_value, str):
            stripped_value = param_value.strip()

            # Handle variable references
            var_match = re.match(r'^\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}$', stripped_value)
            if var_match:
                var_name = var_match.group(1)
                if var_name in context:
                    return context[var_name]
                else:
                    print(f"Warning: Variable '{var_name}' not found in context")
                    # Try to return the original value directly
                    try:
                        return float(stripped_value)
                    except ValueError:
                        return param_value


            if stripped_value in context:
                return context[stripped_value]

            try:
                return float(stripped_value)
            except ValueError:

                if re.match(r'^[-+]?\d*\.?\d+([eE][-+]?\d+)?(\s+[-+]?\d*\.?\d+([eE][-+]?\d+)?)*$', stripped_value):
                    try:
                        values = [float(x) for x in stripped_value.split()]
                        if len(values) == 3:
                            return np.array(values)
                        return values
                    except ValueError:
                        pass
                return stripped_value

        elif isinstance(param_value, list):
            return [self._resolve_parameter_value(item, context) for item in param_value]

        elif isinstance(param_value, dict):
            return {k: self._resolve_parameter_value(v, context) for k, v in param_value.items()}

        return param_value

    def _try_parse_value(self, value_str: str) -> Any:
        """Try to parse a string into a numeric value or an array"""
        value_str = value_str.strip()

        # Try to parse to a floating-point number
        try:
            return float(value_str)
        except ValueError:
            pass

        # Try to parse to an array
        if re.match(r'^[-+]?\d*\.?\d+([eE][-+]?\d+)?(\s+[-+]?\d*\.?\d+([eE][-+]?\d+)?)*$', value_str):
            try:
                values = [float(x) for x in value_str.split()]
                if len(values) == 3:  # 可能是3D向量
                    return np.array(values)
                return values
            except ValueError:
                pass

        return None

    def _execute_atomic_operation(self, op_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Perform individual atomic operations, add parameter parsing and debugging"""
        if op_name not in self.atomic_operations:
            raise ValueError(f"Unknown atomic operation: {op_name}")

        # Parsing parameters
        resolved_params = {}
        for param_name, param_value in parameters.items():
            resolved_value = self._resolve_parameter_value(param_value, context)
            resolved_params[param_name] = resolved_value
            print(f"  Resolved parameter '{param_name}': {type(resolved_value).__name__} = {resolved_value}")

        # Check if the key parameters are valid
        if op_name in ["extract_submesh_by_ratio", "extract_inverse_submesh", "calculate_pivot_point",
                       "separate_extraction"]:
            if "mesh" in resolved_params and isinstance(resolved_params["mesh"], str):
                print(f"Warning: Parameter 'mesh' is a string, not a mesh object")
                # Try to get the correct mesh object from the context
                mesh_name = resolved_params["mesh"]
                if mesh_name in context and hasattr(context[mesh_name], 'vertices'):
                    resolved_params["mesh"] = context[mesh_name]
                    print(f"  Fixed parameter 'mesh' using context[{mesh_name}]")
                else:
                    print(f"  Cannot fix parameter 'mesh', using empty mesh")
                    resolved_params["mesh"] = o3d.geometry.TriangleMesh()

        # Perform the action
        try:
            result = self.atomic_operations[op_name](**resolved_params)
            print(f"  Operation '{op_name}' completed successfully")
            return result
        except Exception as e:
            print(f"Error in atomic operation '{op_name}': {str(e)}")
            print(f"Parameters: {resolved_params}")
            # Returns a default value instead of throwing an exception
            if op_name.startswith("extract"):
                return o3d.geometry.TriangleMesh()
            elif op_name.startswith("calculate"):
                return np.array([0.0, 0.0, 0.0])
            elif op_name == "separate_extraction":
                return (o3d.geometry.TriangleMesh(), o3d.geometry.TriangleMesh())
            else:
                raise

    def _extract_inverse_points(self, pcd: o3d.geometry.PointCloud,
                                reference: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Extract points from a point cloud that are not in the reference point cloud"""
        if pcd is None or not pcd.has_points() or len(pcd.points) == 0:
            return o3d.geometry.PointCloud()

        if reference is None or not reference.has_points() or len(reference.points) == 0:
            return pcd

        points = np.asarray(pcd.points)

        # Gets the boundary of the reference point cloud
        ref_points = np.asarray(reference.points)
        ref_min = np.min(ref_points, axis=0) - 1e-6
        ref_max = np.max(ref_points, axis=0) + 1e-6

        # Find points that are not within the reference point cloud boundary
        not_in_region = ~np.all((points >= ref_min) & (points <= ref_max), axis=1)
        frame_points = points[not_in_region]

        # Create a new point cloud
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(frame_points)

        # If there are normals, extract the corresponding normals as well
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            frame_normals = normals[not_in_region]
            result.normals = o3d.utility.Vector3dVector(frame_normals)

        return result

    def edit_model(self, model_type: str, operation_name: str,
                   input_mesh: o3d.geometry.TriangleMesh,
                   parameters: Dict[str, Any] = None) -> o3d.geometry.TriangleMesh:
        """
        Edit the model

        Args:
            model_type: Model type (e.g., "window")
            operation_name: Action name (such as "open_window")
            input_mesh: Enter the grid
            parameters: User-provided parameters

        Returns:
            Edited grid
        """
        if model_type not in self.configs:
            raise ValueError(f"Model type '{model_type}' not found in configurations")

        model_config = self.configs[model_type]
        operations = model_config.get("operations", {})

        # Find the action (first in specific, then in common)
        operation_config = None
        if "specific" in operations and operation_name in operations["specific"]:
            operation_config = operations["specific"][operation_name]
        elif "common" in operations and operation_name in operations["common"]:
            operation_config = operations["common"][operation_name]

        if operation_config is None:
            raise ValueError(f"Operation '{operation_name}' not found for model type '{model_type}'")


        default_params = {}
        for param_name, param_config in operation_config.get("parameters", {}).items():
            default_params[param_name] = param_config.get("default")

        user_params = parameters or {}
        final_params = {**default_params, **user_params}

        # Initialize the context
        context = {
            "input_mesh": input_mesh,
            **final_params
        }

        print(f"Starting operation '{operation_name}' with parameters: {final_params}")

        # Perform atomic operation sequences
        atomic_ops = operation_config.get("atomic_operations", [])
        for i, atomic_op in enumerate(atomic_ops):
            op_name = atomic_op["operation"]
            op_params = atomic_op["parameters"]
            output_var = atomic_op["output"]

            print(f"Step {i + 1}/{len(atomic_ops)}: {op_name} -> {output_var}")

            try:
                result = self._execute_atomic_operation(op_name, op_params, context)

                # Handling multiple return values (such as separate_extraction return tuples)
                if isinstance(result, (tuple, list)) and isinstance(output_var, (tuple, list)):
                    # 将多个返回值分别存储到上下文中
                    for idx, var_name in enumerate(output_var):
                        if idx < len(result):
                            context[var_name] = result[idx]
                            print(f"  Stored result in context as '{var_name}'")
                else:
                    # Single return value, stored normally
                    context[output_var] = result
                    print(f"  Stored result in context as '{output_var}'")

            except Exception as e:
                raise RuntimeError(f"Error executing atomic operation '{op_name}': {str(e)}")

        # Return the final result
        if "result_mesh" in context:
            return context["result_mesh"]
        else:
            return input_mesh

    def _atomic_extract_submesh_by_inset(self, mesh: o3d.geometry.TriangleMesh,
                                         params: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
        """Atomic Operations: Extract subgrids based on boundary indentation - general, support parameterized indentation scale and axial direction"""
        # Handling point cloud warnings: Rebuild the mesh
        if not mesh.has_triangles():
            print("Warning: No triangle mesh, rebuilding...")
            pcd = o3d.geometry.PointCloud(points=mesh.vertices)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

        bounds = self._atomic_get_bounds(mesh)
        min_bound = bounds["min"]
        max_bound = bounds["max"]
        size = bounds["size"]

        # Get the indentation scale from params (default 0.05, i.e. 5% frame thickness)
        inset_ratio = params.get("inset_ratio", 0.05)
        inset_amount = np.min(size) * inset_ratio
        axes = params.get("inset_axes", [0, 1])  # By default, the X and Y axes are indented, and the Z thickness is preserved

        # Indented boundaries
        inset_min = min_bound.copy()
        inset_max = max_bound.copy()
        for ax in axes:
            inset_min[ax] += inset_amount
            inset_max[ax] -= inset_amount

        # Extract subgrids (filter vertices and triangles)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        in_region = np.all((vertices >= inset_min) & (vertices <= inset_max), axis=1)
        if not np.any(in_region):
            raise ValueError("Submes cannot be extracted, ensuring that the indentation parameters are correct")

        sub_indices = np.where(in_region)[0]
        vertex_map = {old: new for new, old in enumerate(sub_indices)}

        sub_triangles = []
        for tri in triangles:
            if all(in_region[tri[i]] for i in range(3)):
                sub_triangles.append([vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]])

        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = o3d.utility.Vector3dVector(vertices[sub_indices])
        sub_mesh.triangles = o3d.utility.Vector3iVector(sub_triangles) if sub_triangles else o3d.utility.Vector3iVector(
            [])

        print(f"Extract submesh successful: vertices {len(sub_mesh.vertices)}, triangles {len(sub_mesh.triangles)}")
        return sub_mesh

    def _atomic_create_frame_by_subtraction(self, original_mesh: o3d.geometry.TriangleMesh,
                                            sub_mesh: o3d.geometry.TriangleMesh,
                                            params: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
        """Atomic Operation: Subtract subgrids from the original mesh to create a frame - generic, remove using Boolean difference or fallback"""
        if not original_mesh.has_triangles() or not sub_mesh.has_triangles():
            raise ValueError("Inputs or subgrids have no triangles and cannot create frames")

        try:
            # Open3D Boolean difference
            frame = original_mesh.compute_boolean_difference(sub_mesh)
        except AttributeError:
            print("Warning: Boolean is not supported, fallback to remove sub-mesh triangles")
            vertices = np.asarray(original_mesh.vertices)
            triangles = np.asarray(original_mesh.triangles)
            sub_vertices_set = set(tuple(v) for v in np.asarray(sub_mesh.vertices))

            keep_mask = np.array(
                [not all(tuple(vertices[tri[i]]) in sub_vertices_set for i in range(3)) for tri in triangles])
            frame_triangles = triangles[keep_mask]

            frame = o3d.geometry.TriangleMesh()
            frame.vertices = original_mesh.vertices
            frame.triangles = o3d.utility.Vector3iVector(frame_triangles)

        print("Creating a framework is successful")
        return frame



    def _atomic_copy_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Copy mesh"""
        return copy.deepcopy(mesh)

    def _atomic_get_bounds(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, np.ndarray]:
        """Get the grid boundary"""
        vertices = np.asarray(mesh.vertices)
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        return {"min": min_bound, "max": max_bound, "center": center, "size": size}

    def _atomic_translate_mesh(self, mesh: o3d.geometry.TriangleMesh,
                               vector: Union[List[float], np.ndarray]) -> o3d.geometry.TriangleMesh:
        """Pan the mesh"""
        result = copy.deepcopy(mesh)
        result.translate(np.array(vector))
        return result

    def _atomic_rotate_mesh(self, mesh: o3d.geometry.TriangleMesh,
                            angle: float, axis: Union[List[float], np.ndarray]) -> o3d.geometry.TriangleMesh:
        """Rotate the grid (around the origin)"""
        result = copy.deepcopy(mesh)
        axis = np.array(axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.radians(angle))
        result.rotate(rotation_matrix)
        return result

    def _atomic_rotate_mesh_around_pivot(self, mesh: o3d.geometry.TriangleMesh,
                                         angle: float, axis: Union[List[float], np.ndarray],
                                         pivot: Union[List[float], np.ndarray]) -> o3d.geometry.TriangleMesh:
        """Rotate the mesh around the pivot point"""
        result = copy.deepcopy(mesh)
        pivot = np.array(pivot)
        axis = np.array(axis)

        # Pan to the origin point, rotate, and pan back again
        result.translate(-pivot)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.radians(angle))
        result.rotate(rotation_matrix)
        result.translate(pivot)

        return result

    def _atomic_scale_mesh(self, mesh: o3d.geometry.TriangleMesh,
                           factor: float, center: Union[List[float], np.ndarray]) -> o3d.geometry.TriangleMesh:
        """Scale the grid"""
        result = copy.deepcopy(mesh)
        center = np.array(center)

        # Pan to origin, zoom, and pan back
        result.translate(-center)
        result.scale(factor, np.array([0, 0, 0]))
        result.translate(center)

        return result

    def _atomic_combine_meshes(self, meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
        """Merge multiple meshes"""
        if not meshes:
            return o3d.geometry.TriangleMesh()

        result = copy.deepcopy(meshes[0])
        for i in range(1, len(meshes)):
            result += copy.deepcopy(meshes[i])

        return result

    def _atomic_extract_submesh_by_ratio(self, mesh: o3d.geometry.TriangleMesh,
                                         ratio: float, frame_thickness_ratio: float,
                                         based_on: str = "largest_face") -> o3d.geometry.TriangleMesh:
        """Extract subgrids proportionally"""
        # Make sure the input is a valid mesh
        if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print("Warning: Invalid mesh provided to extract_submesh_by_ratio")
            return o3d.geometry.TriangleMesh()

        # Check if the grid has triangles
        if not mesh.has_triangles() or len(mesh.triangles) == 0:
            print("Warning: Mesh has no triangles, treating as point cloud")
            # For point clouds, use the new point cloud extraction method
            return self._extract_points_by_ratio(mesh, ratio, frame_thickness_ratio, based_on)

        bounds = self._atomic_get_bounds(mesh)
        min_bound = bounds["min"]
        max_bound = bounds["max"]
        size = bounds["size"]


        if based_on == "largest_face":

            areas = [size[0] * size[1], size[0] * size[2], size[1] * size[2]]
            main_plane = np.argmax(areas)
        else:
            main_plane = 0  # Default XY plane

        frame_thickness = min(size) * frame_thickness_ratio

        sash_min = min_bound + frame_thickness
        sash_max = max_bound - frame_thickness

        if main_plane == 0:
            sash_size = sash_max - sash_min
            reduction = (1 - ratio) * sash_size / 2
            sash_min[0] += reduction[0]
            sash_max[0] -= reduction[0]
            sash_min[1] += reduction[1]
            sash_max[1] -= reduction[1]
        elif main_plane == 1:
            sash_size = sash_max - sash_min
            reduction = (1 - ratio) * sash_size / 2
            sash_min[0] += reduction[0]
            sash_min[2] += reduction[2]
            sash_max[0] -= reduction[0]
            sash_max[2] -= reduction[2]
        else:
            sash_size = sash_max - sash_min
            reduction = (1 - ratio) * sash_size / 2
            sash_min[1] += reduction[1]
            sash_min[2] += reduction[2]
            sash_max[1] -= reduction[1]
            sash_max[2] -= reduction[2]

        # Extract subgrids
        return self._extract_submesh_by_bounds(mesh, sash_min, sash_max)

    def _extract_points_by_ratio(self, pcd: o3d.geometry.PointCloud,
                                 ratio: float, frame_thickness_ratio: float,
                                 based_on: str = "largest_face") -> o3d.geometry.PointCloud:
        """Extract points proportionally from point clouds - Used to work with point clouds without triangles"""
        if pcd is None or not pcd.has_points() or len(pcd.points) == 0:
            return o3d.geometry.PointCloud()

        points = np.asarray(pcd.points)


        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        size = max_bound - min_bound
        center = (min_bound + max_bound) / 2


        if based_on == "largest_face":
            areas = [size[0] * size[1], size[0] * size[2], size[1] * size[2]]
            main_plane = np.argmax(areas)
        else:
            main_plane = 0


        frame_thickness = min(size) * frame_thickness_ratio
        sash_min = min_bound + frame_thickness
        sash_max = max_bound - frame_thickness


        if main_plane == 0:
            sash_size = sash_max - sash_min
            reduction = (1 - ratio) * sash_size / 2
            sash_min[0] += reduction[0]
            sash_max[0] -= reduction[0]
            sash_min[1] += reduction[1]
            sash_max[1] -= reduction[1]
        elif main_plane == 1:
            sash_size = sash_max - sash_min
            reduction = (1 - ratio) * sash_size / 2
            sash_min[0] += reduction[0]
            sash_min[2] += reduction[2]
            sash_max[0] -= reduction[0]
            sash_max[2] -= reduction[2]
        else:
            sash_size = sash_max - sash_min
            reduction = (1 - ratio) * sash_size / 2
            sash_min[1] += reduction[1]
            sash_min[2] += reduction[2]
            sash_max[1] -= reduction[1]
            sash_max[2] -= reduction[2]


        in_region = np.all((points >= sash_min) & (points <= sash_max), axis=1)
        sash_points = points[in_region]


        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(sash_points)


        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            sash_normals = normals[in_region]
            result.normals = o3d.utility.Vector3dVector(sash_normals)

        return result

    def _extract_submesh_by_bounds(self, mesh: o3d.geometry.TriangleMesh,
                                   min_bound: np.ndarray, max_bound: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Extract submeshes by boundary"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)


        in_region = np.all((vertices >= min_bound) & (vertices <= max_bound), axis=1)


        valid_triangles = []
        for tri in triangles:
            if in_region[tri[0]] and in_region[tri[1]] and in_region[tri[2]]:
                valid_triangles.append(tri)

        if not valid_triangles:
            return o3d.geometry.TriangleMesh()


        result = o3d.geometry.TriangleMesh()
        result.vertices = o3d.utility.Vector3dVector(vertices)
        result.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))

        return result

    def _atomic_separate_extraction(self, mesh: o3d.geometry.TriangleMesh,
                                    ratio: float, frame_thickness_ratio: float,
                                    based_on: str = "largest_face") -> Tuple[
        o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """Split interception policy"""

        if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print("Warning: Invalid mesh provided to separate_extraction")
            return o3d.geometry.TriangleMesh(), o3d.geometry.TriangleMesh()


        if not mesh.has_triangles() or len(mesh.triangles) == 0:
            print("Warning: Mesh has no triangles, treating as point cloud")

            return self._separate_extraction_points(mesh, ratio, frame_thickness_ratio, based_on)
        else:

            return self._separate_extraction_mesh(mesh, ratio, frame_thickness_ratio, based_on)

    def _separate_extraction_points(self, pcd: o3d.geometry.TriangleMesh,
                                    ratio: float, frame_thickness_ratio: float,
                                    based_on: str = "largest_face") -> Tuple[
        o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """Split interception strategy: Handle point clouds (meshes without triangles)"""
        # Convert TriangleMesh to PointCloud
        points = np.asarray(pcd.vertices)

        # Create a PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # If there are normals, copy them as well
        if pcd.has_vertex_normals():
            normals = np.asarray(pcd.vertex_normals)
            point_cloud.normals = o3d.utility.Vector3dVector(normals)


        sash_points = self._extract_points_by_ratio(point_cloud, ratio, frame_thickness_ratio, based_on)


        frame_points = self._extract_inverse_points(point_cloud, sash_points)


        frame_mesh = o3d.geometry.TriangleMesh()
        frame_mesh.vertices = frame_points.points
        if frame_points.has_normals():
            frame_mesh.vertex_normals = frame_points.normals

        sash_mesh = o3d.geometry.TriangleMesh()
        sash_mesh.vertices = sash_points.points
        if sash_points.has_normals():
            sash_mesh.vertex_normals = sash_points.normals

        return frame_mesh, sash_mesh

    def _separate_extraction_mesh(self, mesh: o3d.geometry.TriangleMesh,
                                  ratio: float, frame_thickness_ratio: float,
                                  based_on: str = "largest_face") -> Tuple[
        o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """Split intercept strategy: Handles meshes with triangles"""

        mesh_copy = copy.deepcopy(mesh)


        sash_mesh = self._atomic_extract_submesh_by_ratio(mesh_copy, ratio, frame_thickness_ratio, based_on)


        frame_mesh = self._atomic_extract_inverse_submesh(mesh_copy, sash_mesh)

        return frame_mesh, sash_mesh

    def _atomic_extract_inverse_submesh(self, mesh: o3d.geometry.TriangleMesh,
                                        reference: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Extract parts that are not in the reference grid"""

        if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print("Warning: Invalid mesh provided to extract_inverse_submesh")
            return o3d.geometry.TriangleMesh()

        if reference is None or not hasattr(reference, 'vertices') or len(reference.vertices) == 0:
            print("Warning: Invalid reference mesh provided to extract_inverse_submesh")
            return mesh


        if not mesh.has_triangles() or len(mesh.triangles) == 0:
            print("Warning: Mesh has no triangles, treating as point cloud")

            return self._extract_inverse_points(mesh, reference)


        ref_bounds = self._atomic_get_bounds(reference)
        ref_min = ref_bounds["min"] - 1e-6
        ref_max = ref_bounds["max"] + 1e-6

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)


        not_in_region = ~np.all((vertices >= ref_min) & (vertices <= ref_max), axis=1)


        valid_triangles = []
        for tri in triangles:
            if not_in_region[tri[0]] and not_in_region[tri[1]] and not_in_region[tri[2]]:
                valid_triangles.append(tri)

        if not valid_triangles:
            return o3d.geometry.TriangleMesh()

        # 创建新网格
        result = o3d.geometry.TriangleMesh()
        result.vertices = o3d.utility.Vector3dVector(vertices)
        result.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))

        return result

    def _atomic_split_mesh_by_axis(self, mesh: o3d.geometry.TriangleMesh,
                                   axis: str, ratio: float) -> List[o3d.geometry.TriangleMesh]:
        """Divide the grid by axis"""
        bounds = self._atomic_get_bounds(mesh)
        min_bound = bounds["min"]
        max_bound = bounds["max"]

        if axis == "x":
            split_value = min_bound[0] + (max_bound[0] - min_bound[0]) * ratio
            left_min = min_bound
            left_max = np.array([split_value, max_bound[1], max_bound[2]])
            right_min = np.array([split_value, min_bound[1], min_bound[2]])
            right_max = max_bound
        elif axis == "y":
            split_value = min_bound[1] + (max_bound[1] - min_bound[1]) * ratio
            left_min = min_bound
            left_max = np.array([max_bound[0], split_value, max_bound[2]])
            right_min = np.array([min_bound[0], split_value, min_bound[2]])
            right_max = max_bound
        elif axis == "z":
            split_value = min_bound[2] + (max_bound[2] - min_bound[2]) * ratio
            left_min = min_bound
            left_max = np.array([max_bound[0], max_bound[1], split_value])
            right_min = np.array([min_bound[0], min_bound[1], split_value])
            right_max = max_bound
        else:
            raise ValueError(f"Invalid axis: {axis}")

        left_mesh = self._extract_submesh_by_bounds(mesh, left_min, left_max)
        right_mesh = self._extract_submesh_by_bounds(mesh, right_min, right_max)

        return [left_mesh, right_mesh]

    def _atomic_calculate_translation_vector(self, direction: str,
                                             bounds: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate the translation vector"""
        size = bounds["size"]

        if direction == "right":
            return np.array([size[0], 0.0, 0.0])
        elif direction == "left":
            return np.array([-size[0], 0.0, 0.0])
        elif direction == "up":
            return np.array([0.0, size[1], 0.0])
        elif direction == "down":
            return np.array([0.0, -size[1], 0.0])
        elif direction == "front":
            return np.array([0.0, 0.0, size[2]])
        elif direction == "back":
            return np.array([0.0, 0.0, -size[2]])
        else:
            return np.array([0.0, 0.0, 0.0])

    def _atomic_calculate_pivot_point(self, mesh: o3d.geometry.TriangleMesh,
                                      pivot_type: str, edge: str = None,
                                      frame_thickness_ratio: float = 0.05) -> np.ndarray:
        """Calculate pivot points"""
        # 确保输入是有效的网格
        if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print("Warning: Invalid mesh provided to calculate_pivot_point")
            return np.array([0.0, 0.0, 0.0])

        bounds = self._atomic_get_bounds(mesh)
        min_bound = bounds["min"]
        max_bound = bounds["max"]
        center = bounds["center"]
        size = bounds["size"]

        if pivot_type == "center":
            return center
        elif pivot_type == "edge" and edge:

            areas = [size[0] * size[1], size[0] * size[2], size[1] * size[2]]
            main_plane = np.argmax(areas)

            if main_plane == 0:
                if edge == "left":
                    return np.array([min_bound[0], center[1], center[2]])
                elif edge == "right":
                    return np.array([max_bound[0], center[1], center[2]])
                elif edge == "top":
                    return np.array([center[0], max_bound[1], center[2]])
                elif edge == "bottom":
                    return np.array([center[0], min_bound[1], center[2]])
            elif main_plane == 1:
                if edge == "left":
                    return np.array([min_bound[0], center[1], center[2]])
                elif edge == "right":
                    return np.array([max_bound[0], center[1], center[2]])
                elif edge == "top":
                    return np.array([center[0], center[1], max_bound[2]])
                elif edge == "bottom":
                    return np.array([center[0], center[1], min_bound[2]])
            else:
                if edge == "left":
                    return np.array([center[0], min_bound[1], center[2]])
                elif edge == "right":
                    return np.array([center[0], max_bound[1], center[2]])
                elif edge == "top":
                    return np.array([center[0], center[1], max_bound[2]])
                elif edge == "bottom":
                    return np.array([center[0], center[1], min_bound[2]])

        return center

    def _align_sash_to_frame(self, sash_mesh: o3d.geometry.TriangleMesh, frame_mesh: o3d.geometry.TriangleMesh,
                             edge: str) -> o3d.geometry.TriangleMesh:
        """Based on local edge point coincidences"""


        sash_bounds = self._atomic_get_bounds(sash_mesh)
        frame_bounds = self._atomic_get_bounds(frame_mesh)


        sash_size = sash_bounds["size"]
        areas = [sash_size[0] * sash_size[1], sash_size[0] * sash_size[2], sash_size[1] * sash_size[2]]
        main_plane = np.argmax(areas)


        sash_points = np.asarray(sash_mesh.vertices)
        frame_points = np.asarray(frame_mesh.vertices)


        edge_threshold = 0.05 * np.min(sash_size)


        if edge == "left":
            if main_plane == 0:  # XY
                sash_edge_points = sash_points[np.abs(sash_points[:, 0] - sash_bounds["min"][0]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 0] - frame_bounds["min"][0]) < edge_threshold]
            elif main_plane == 1:  # XZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 0] - sash_bounds["min"][0]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 0] - frame_bounds["min"][0]) < edge_threshold]
            else:  # YZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 1] - sash_bounds["min"][1]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 1] - frame_bounds["min"][1]) < edge_threshold]
        elif edge == "right":
            if main_plane == 0:  # XY
                sash_edge_points = sash_points[np.abs(sash_points[:, 0] - sash_bounds["max"][0]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 0] - frame_bounds["max"][0]) < edge_threshold]
            elif main_plane == 1:  # XZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 0] - sash_bounds["max"][0]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 0] - frame_bounds["max"][0]) < edge_threshold]
            else:  # YZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 1] - sash_bounds["max"][1]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 1] - frame_bounds["max"][1]) < edge_threshold]
        elif edge == "top":
            if main_plane == 0:  # XY
                sash_edge_points = sash_points[np.abs(sash_points[:, 1] - sash_bounds["max"][1]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 1] - frame_bounds["max"][1]) < edge_threshold]
            elif main_plane == 1:  # XZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 2] - sash_bounds["max"][2]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 2] - frame_bounds["max"][2]) < edge_threshold]
            else:  # YZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 2] - sash_bounds["max"][2]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 2] - frame_bounds["max"][2]) < edge_threshold]
        elif edge == "bottom":
            if main_plane == 0:  # XY
                sash_edge_points = sash_points[np.abs(sash_points[:, 1] - sash_bounds["min"][1]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 1] - frame_bounds["min"][1]) < edge_threshold]
            elif main_plane == 1:  # XZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 2] - sash_bounds["min"][2]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 2] - frame_bounds["min"][2]) < edge_threshold]
            else:  # YZ
                sash_edge_points = sash_points[np.abs(sash_points[:, 2] - sash_bounds["min"][2]) < edge_threshold]
                frame_edge_points = frame_points[np.abs(frame_points[:, 2] - frame_bounds["min"][2]) < edge_threshold]
        else:
            raise ValueError(f"Unsupported edge: {edge}")


        if len(sash_edge_points) == 0 or len(frame_edge_points) == 0:
            raise ValueError("If you can't extract edge points, check the mesh density or threshold")

        sash_edge_mean = np.mean(sash_edge_points, axis=0)
        frame_edge_mean = np.mean(frame_edge_points, axis=0)


        translation = frame_edge_mean - sash_edge_mean


        sash_mesh.translate(translation)


        aligned_sash_edge_mean = np.mean(np.asarray(sash_mesh.vertices)[np.abs(
            np.asarray(sash_mesh.vertices)[:, 0] - sash_bounds["min"][0] + translation[0]) < edge_threshold],
                                         axis=0)
        if np.linalg.norm(aligned_sash_edge_mean - frame_edge_mean) > 1e-3:
            print("Warning: 边缘重合度未达预期，可能需调整阈值")

        return sash_mesh

    def _atomic_calculate_rotation_axis(self, mesh: o3d.geometry.TriangleMesh,
                                        edge: str) -> np.ndarray:
        """Calculate the axis of rotation - Make sure the axis of rotation is oriented correctly"""

        bounds = self._atomic_get_bounds(mesh)
        size = bounds["size"]


        areas = [size[0] * size[1], size[0] * size[2], size[1] * size[2]]
        main_plane = np.argmax(areas)


        if main_plane == 0:
            if edge in ["left", "right"]:
                return np.array([0.0, 0.0, 1.0])  # Z轴
            elif edge in ["top", "bottom"]:
                return np.array([1.0, 0.0, 0.0])  # X轴
        elif main_plane == 1:
            if edge in ["left", "right"]:
                return np.array([0.0, 1.0, 0.0])  # Y轴
            elif edge in ["top", "bottom"]:
                return np.array([1.0, 0.0, 0.0])  # X轴
        else:
            if edge in ["left", "right"]:
                return np.array([0.0, 1.0, 0.0])  # Y轴
            elif edge in ["top", "bottom"]:
                return np.array([0.0, 0.0, 1.0])  # Z轴

        return np.array([0.0, 0.0, 1.0])

    def _atomic_calculate_scale_center(self, mesh: o3d.geometry.TriangleMesh,
                                       center_type: str) -> np.ndarray:
        """Compute the zoom center"""
        if center_type == "origin":
            return np.array([0.0, 0.0, 0.0])
        else:  # center
            bounds = self._atomic_get_bounds(mesh)
            return bounds["center"]

    def _atomic_calculate_slide_direction(self, mesh: o3d.geometry.TriangleMesh,
                                          based_on: str = "largest_face") -> np.ndarray:
        """Calculate the sliding direction"""
        bounds = self._atomic_get_bounds(mesh)
        size = bounds["size"]

        if based_on == "largest_face":

            areas = [size[0] * size[1], size[0] * size[2], size[1] * size[2]]
            main_plane = np.argmax(areas)

            if main_plane == 0:
                return np.array([1.0, 0.0, 0.0])
            elif main_plane == 1:
                return np.array([1.0, 0.0, 0.0])
            else:
                return np.array([0.0, 1.0, 0.0])
        else:
            return np.array([1.0, 0.0, 0.0])

    def _atomic_calculate_slide_distance(self, mesh: o3d.geometry.TriangleMesh,
                                         percent: float,
                                         based_on: str = "largest_face") -> float:
        """Calculate the slip distance"""
        bounds = self._atomic_get_bounds(mesh)
        size = bounds["size"]

        if based_on == "largest_face":

            areas = [size[0] * size[1], size[0] * size[2], size[1] * size[2]]
            main_plane = np.argmax(areas)

            if main_plane == 0:
                move_len = size[0] / 2
            elif main_plane == 1:
                move_len = size[0] / 2
            else:
                move_len = size[1] / 2
        else:
            move_len = size[0] / 2

        return (percent / 100.0) * move_len

    def _atomic_adjust_angle_by_direction(self, angle: float, direction: str) -> float:
        """Adjust the angle according to the direction"""
        if direction == "inward":
            return -angle
        else:
            return angle

    def _atomic_reset_mesh_position(self, mesh: o3d.geometry.TriangleMesh,
                                    reference: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Resets the mesh position to the original position of the reference mesh"""

        ref_bounds = self._atomic_get_bounds(reference)
        mesh_bounds = self._atomic_get_bounds(mesh)


        offset = ref_bounds["center"] - mesh_bounds["center"]


        result = copy.deepcopy(mesh)
        result.translate(offset)

        return result

def parse_edit_command(command: str, model_type: str) -> Tuple[str, Dict]:
    """
    Parse the edit command and return the operation name and parameters
    """
    command_lower = command.lower()


    if model_type.lower() == "window":
        if any(word in command_lower for word in ["打开", "开启", "open"]):
            if any(word in command_lower for word in ["平移", "滑动", "slide"]):

                percent_match = re.search(r'(\d+)%', command)
                percent = float(percent_match.group(1)) if percent_match else 80.0
                return "slide_open_window", {"percent_open": percent}
            else:

                angle_match = re.search(r'(\d+)\s*(度|°|degree)', command)
                angle = float(angle_match.group(1)) if angle_match else 90.0

                direction = "outward"
                if any(word in command_lower for word in ["向内", "inward", "inside"]):
                    direction = "inward"

                pivot = "left"
                if any(word in command_lower for word in ["右边", "右侧", "right"]):
                    pivot = "right"
                elif any(word in command_lower for word in ["上边", "顶部", "top"]):
                    pivot = "top"
                elif any(word in command_lower for word in ["下边", "底部", "bottom"]):
                    pivot = "bottom"

                return "open_window", {"angle": angle, "direction": direction, "pivot_edge": pivot}

        elif any(word in command_lower for word in ["关闭", "合上", "close"]):
            return "close_window", {}

        elif any(word in command_lower for word in ["添加", "增加", "add"]):
            direction = "right"
            if any(word in command_lower for word in ["左", "left"]):
                direction = "left"
            elif any(word in command_lower for word in ["上", "up"]):
                direction = "up"
            elif any(word in command_lower for word in ["下", "down"]):
                direction = "down"
            elif any(word in command_lower for word in ["前", "front"]):
                direction = "front"
            elif any(word in command_lower for word in ["后", "back"]):
                direction = "back"

            return "add", {"direction": direction}
        else:

            return "open_window", {"angle": 90.0, "direction": "outward", "pivot_edge": "left"}
    else:

        return "", {}


# Usage examples
if __name__ == "__main__":

    editor = UniversalModelEditor("./models/model_configs")

    window_mesh = o3d.io.read_triangle_mesh("window.ply")

    opened_window = editor.edit_model(
        "window", "open_window", window_mesh,
        {"angle": 75.0, "pivot_edge": "left", "direction": "outward"}
    )

    slided_window = editor.edit_model(
        "window", "slide_open_window", window_mesh,
        {"percent_open": 80.0}
    )

    added_window = editor.edit_model(
        "window", "add", window_mesh,
        {"direction": "right"}
    )