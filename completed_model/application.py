import streamlit as st
import os
import shutil
import time
import pyvista as pv
import numpy as np
from PIL import Image
from stpyvista import stpyvista
import main
from edit_model import UniversalModelEditor, parse_edit_command
import open3d as o3d
from window_editor import SemanticWindowEditor


st.set_page_config(layout="wide", page_title="3D model generation and editing system")


def init_directories():
    os.makedirs("./test_image", exist_ok=True)
    os.makedirs("./output/output_latest", exist_ok=True)
    os.makedirs("./output/edited_input", exist_ok=True)
    os.makedirs("./output/edited_output", exist_ok=True)
    os.makedirs("./temp", exist_ok=True)


def save_uploaded_image(uploaded_file):
    image_path = "./test_image/1.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return image_path

def run_3d_reconstruction(image_path, fit_strength):
    # Run the main processing flow
    success, class_name = main.main(image_path, fit_strength=fit_strength)

    latest_dir = "./output/output_latest"
    os.makedirs(latest_dir, exist_ok=True)

    mesh_path = os.path.join(latest_dir, f"{class_name}.ply")
    if not os.path.exists(mesh_path):

        possible_files = [
            os.path.join(latest_dir, "reconstructed_mesh.ply"),
            os.path.join(latest_dir, "post_fit.ply"),
            os.path.join(latest_dir, "aligned.ply")
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):

                shutil.copy(file_path, mesh_path)
                print(f"Locate and copy the model file: {file_path} -> {mesh_path}")
                break

    return success, class_name


# Perform semantic editing
def run_semantic_edit(class_name, edit_command):
    input_path = os.path.join("./output/edited_input", f"{class_name}.ply")
    output_path = os.path.join("./output/edited_output", f"edited_{class_name}.ply")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if class_name.lower() == "window":
        # Use the Universal Editor
        editor = UniversalModelEditor("./models/model_configs")

        try:
            mesh = o3d.io.read_triangle_mesh(input_path)
            if not mesh.has_vertices():
                st.error(f"Unable to load input mesh: {input_path}")
                return False

            # Parse the edit command
            operation_name, parameters = parse_edit_command(edit_command, class_name)

            if not operation_name:
                st.error(f"Unable to parse edit commands: {edit_command}")
                return False

            # Executive editing
            edited_mesh = editor.edit_model(class_name, operation_name, mesh, parameters)


            if not edited_mesh.has_triangles() or len(edited_mesh.triangles) == 0:

                pcd = o3d.geometry.PointCloud()
                pcd.points = edited_mesh.vertices
                if edited_mesh.has_vertex_normals():
                    pcd.normals = edited_mesh.vertex_normals
                o3d.io.write_point_cloud(output_path, pcd)
            else:

                o3d.io.write_triangle_mesh(output_path, edited_mesh)

            return True

        except Exception as e:
            st.error(f"Editing failed: {str(e)}")
            return False
    else:
        st.warning(f"Semantic editing of the {class_name} category is not supported at this time")
        return False


# Visualize 3D models
def visualize_3d_model(model_path):
    """Visualize 3D models"""
    try:

        if not os.path.exists(model_path):
            st.error(f"The model file does not exist: {model_path}")
            return None

        # Try using Open3D reading
        try:

            mesh = o3d.io.read_triangle_mesh(model_path)

            if not mesh.has_vertices():
                st.error("The model has no vertices")
                return None

            # Convert to PyVista format
            vertices = np.asarray(mesh.vertices)

            plotter = pv.Plotter(window_size=[400, 400], off_screen=True)

            if mesh.has_triangles() and len(mesh.triangles) > 0:

                faces = np.asarray(mesh.triangles)

                faces = np.insert(faces, 0, 3, axis=1)
                cloud = pv.PolyData(vertices, faces)

                plotter.add_mesh(
                    cloud,
                    color='lightblue',
                    show_edges=True,
                    edge_color='gray'
                )
            else:

                cloud = pv.PolyData(vertices)
                plotter.add_points(
                    cloud,
                    color='lightblue',
                    point_size=5,
                    render_points_as_spheres=True
                )

            plotter.view_isometric()
            return plotter

        except Exception as e:
            st.error(f"Open3D read failed: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Model visualization fails: {str(e)}")
        return None

def ensure_directory_exists(path):
    """Make sure the directory exists"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Create a catalog: {directory}")

# Main application
def main_app():
    st.title("3D model generation and editing system")
    init_directories()

    # Initialize the session state
    if 'class_name' not in st.session_state:
        st.session_state.class_name = ""
    if 'model_generated' not in st.session_state:
        st.session_state.model_generated = False

    # Part 1: 3D model generation
    st.header("1. Generate 3D models")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        fit_strength = st.slider("Fitting strength", 0.0, 1.0, 0.5, key="fit_strength")

        if st.button("Generate 3D models", key="generate_btn"):
            if uploaded_file is not None:
                with st.spinner("Processing, please wait..."):

                    image_path = save_uploaded_image(uploaded_file)

                    st.image(uploaded_file, caption="Uploaded images", use_column_width=True)

                    success, class_name = run_3d_reconstruction(image_path, fit_strength)

                    if success:
                        st.session_state.model_generated = True
                        st.session_state.class_name = class_name
                        st.success(f"Successfully generate {class_name} 3D models!")
                    else:
                        st.error("3D model generation failed")
            else:
                st.warning("Please upload your image first")


    if st.session_state.model_generated:
        class_name = st.session_state.class_name

        st.subheader("Generate results")


        model_path = os.path.join("./output/output_latest", f"{class_name}.ply")
        if os.path.exists(model_path):
            st.subheader("3D model")
            plotter = visualize_3d_model(model_path)
            if plotter:
                stpyvista(plotter, key="generated_model")
        else:
            st.warning(f"Model file not found: {model_path}")

        # Displays multi-view images
        st.subheader("Multi-view image")

        # Create tabs to show different perspectives
        tab_names = ["Front view", "Back view", "Left view", "Right view", "Top view", "bottom view"]
        angles = ["front", "back", "side_one", "side_two", "top", "bottom"]
        tabs = st.tabs(tab_names)

        for i, tab in enumerate(tabs):
            with tab:
                col1, col2 = st.columns(2)
                angle = angles[i]
                img_path = os.path.join("./output/output_latest", f"{angle}_view.png")
                normal_path = os.path.join("./output/output_latest", f"{angle}_normal.png")

                if os.path.exists(img_path):
                    col1.image(img_path, caption=f"{tab_names[i]}", use_column_width=True)
                else:
                    col1.warning(f"{tab_names[i]} image not found")

                if os.path.exists(normal_path):
                    col2.image(normal_path, caption=f"{tab_names[i]} normal diagram", use_column_width=True)
                else:
                    col2.warning(f"{tab_names[i]} normal diagram not found")

    # Part 2: Semantic editing
    st.header("2. Semantic editing")

    col1, col2 = st.columns([3, 1])

    with col1:
        edit_command = st.text_input("Edit commands", placeholder="For example: 'Pan window to the right'", key="edit_cmd")

        if st.button("Executive editing", key="edit_btn"):
            if not st.session_state.model_generated:
                st.warning("Please make a 3D model")
            elif not edit_command:
                st.warning("Please enter the edit command")
            else:
                with st.spinner("Editing, please wait..."):
                    class_name = st.session_state.class_name


                    edited_output_dir = "./output/edited_output"
                    ensure_directory_exists(edited_output_dir)

                    success = run_semantic_edit(class_name, edit_command)

                    if success:
                        st.success("Semantic editing success!")


                        edited_path = os.path.join("./output/edited_output", f"edited_{class_name}.ply")
                        if os.path.exists(edited_path):
                            st.subheader("Edited model")
                            plotter = visualize_3d_model(edited_path)
                            if plotter:
                                stpyvista(plotter, key="edited_model")
                        else:
                            st.warning(f"Edited model file not found: {edited_path}")
                    else:
                        st.error("Semantic editing failed")

    with col2:
        st.subheader("Edit examples")
        st.markdown("""
        **Windows edit command:**
        - Add a new window
        - Rotate the window 45 degrees
        - Panning opens 80% of the windows
        - Close the windows
        """)


if __name__ == "__main__":
    main_app()