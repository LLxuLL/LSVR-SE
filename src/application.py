#!/usr/bin/env python3

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
import sys
import time
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import threading
import queue

import streamlit as st
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from open3d.visualization.__main__ import args
from plotly.subplots import make_subplots

# Import LSVR-SE component
from lsvr_se_config import DEFAULT_CONFIG, FAST_CONFIG
from reasoning import LSVRSEReasoning

# Configure Streamlit page
st.set_page_config(
    page_title="LSVR-SE 3D Reconstruction and Editing System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSVR-SE-Web")

# Global variables
REASONING_ENGINE = None
PROCESSING_QUEUE = queue.Queue()
RESULTS_CACHE = {}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #f0fff0;
        border-left: 4px solid #2ca02c;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fffacd;
        border-left: 4px solid #ffd700;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .error-box {
        background-color: #ffe4e1;
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .progress-container {
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .file-info {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .stDownloadButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_engine():
    """Initialize inference engine"""
    global REASONING_ENGINE

    if REASONING_ENGINE is None:
        try:
            with st.spinner("🚀 Initializing LSVR-SE system..."):
                config = FAST_CONFIG  # Use Quick Setup
                REASONING_ENGINE = LSVRSEReasoning(config)
                REASONING_ENGINE.initialize()
                st.success("✅ System initialization complete!")
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            st.stop()


def save_uploaded_file(uploaded_file, temp_dir: str) -> str:
    """Save the uploaded file"""
    if uploaded_file is None:
        return None

    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def process_single_image(image_path: str, text_instruction: str, progress_bar: st.progress):
    """Process a single image"""
    if REASONING_ENGINE is None:
        return None

    try:
        # Update progress
        progress_bar.progress(0.1, "Initializing...")

        # Process image
        progress_bar.progress(0.3, "Performing HSDE feature extraction...")

        progress_bar.progress(0.6, "Performing LC-NeRF 3D reconstruction...")

        progress_bar.progress(0.9, "Performing DPEE semantic editing...")

        # Perform reasoning
        results = REASONING_ENGINE.process_single_image(image_path, text_instruction)

        progress_bar.progress(1.0, "Processing complete!")

        return results

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        progress_bar.progress(1.0, "Processing failed")
        return {'success': False, 'error': str(e)}


def visualize_3d_mesh(mesh: o3d.geometry.TriangleMesh) -> go.Figure:
    """Visualize 3D Mesh"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color='lightblue',
            opacity=0.8,
            flatshading=True
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title="3D Reconstruction Result",
        width=600,
        height=600
    )

    return fig


def display_results(results: Dict[str, Any]):
    """Display processing results"""
    if not results.get('success'):
        st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        return

    # Success message
    st.success(f"✅ Processing complete! Time taken: {results['processing_time']:.2f} seconds")

    # Results display
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Input Image")
        if 'input_image' in results:
            image = Image.open(results['input_image'])
            st.image(image, use_column_width=True)

    with col2:
        st.subheader("🎯 3D Reconstruction Result")
        if 'final_mesh' in results and results['final_mesh'] is not None:
            fig = visualize_3d_mesh(results['final_mesh'])
            st.plotly_chart(fig, use_container_width=True)

    # Detailed information
    with st.expander("📊 Detailed Information"):
        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric("HSDE Features Count", len(results.get('hsde_results', {}).get('semantic_features', [])))

        with col4:
            if 'final_mesh' in results and results['final_mesh'] is not None:
                vertices = len(results['final_mesh'].vertices)
                triangles = len(results['final_mesh'].triangles)
                st.metric("Mesh Vertices", vertices)
                st.metric("Mesh Faces", triangles)

        with col5:
            st.metric("Processing Time", f"{results['processing_time']:.2f}s")

    # Download options
    if 'final_mesh' in results and results['final_mesh'] is not None:
        st.subheader("⬇️ Download Results")

        # Save temporary file
        temp_dir = tempfile.mkdtemp()
        mesh_path = os.path.join(temp_dir, "result.ply")
        o3d.io.write_triangle_mesh(mesh_path, results['final_mesh'])

        with open(mesh_path, "rb") as f:
            mesh_bytes = f.read()

        st.download_button(
            label="Download 3D Model (PLY format)",
            data=mesh_bytes,
            file_name="lsvr_se_result.ply",
            mime="application/octet-stream"
        )

        # Clean up temporary files
        shutil.rmtree(temp_dir)


def main_page():
    """Main page"""
    st.markdown('<h1 class="main-header">🎨 LSVR-SE 3D Reconstruction and Editing System</h1>', unsafe_allow_html=True)

    # Initialize system
    initialize_engine()

    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Single Image Processing", "Batch Processing", "Interactive Editing", "Results Viewer"],
        index=0
    )

    if mode == "Single Image Processing":
        single_image_page()
    elif mode == "Batch Processing":
        batch_processing_page()
    elif mode == "Interactive Editing":
        interactive_editing_page()
    elif mode == "Results Viewer":
        results_viewer_page()


def single_image_page():
    """Single image processing page"""
    st.markdown('<h2 class="section-header">📸 Single Image Processing</h2>', unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload image file",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Supports PNG, JPG, JPEG, BMP image formats"
    )

    # Text instruction input
    text_instruction = st.text_area(
        "Enter editing instruction (optional)",
        placeholder="Example: Add a window on the wall",
        height=100,
        help="Enter natural language editing instruction"
    )

    # Process button
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                # Create temporary file
                temp_dir = tempfile.mkdtemp()
                image_path = save_uploaded_file(uploaded_file, temp_dir)

                if image_path:
                    # Progress bar
                    progress_bar = st.progress(0)

                    # Process image
                    results = process_single_image(image_path, text_instruction, progress_bar)

                    # Display results
                    if results:
                        display_results(results)

                    # Clean up temporary files
                    shutil.rmtree(temp_dir)

    # Example showcase
    with st.expander("📚 Usage Examples"):
        st.markdown("""
        ### Basic Usage
        1. **Upload image**: Select an image containing architectural elements
        2. **Enter instruction**: Describe the editing operation you want
        3. **Start processing**: Click the process button to start 3D reconstruction

        ### Supported Editing Instructions
        - **Add elements**: "Add window", "Add door", "Add column"
        - **Remove elements**: "Remove door", "Delete window"
        - **Transform operations**: "Rotate 45 degrees", "Scale 1.5x", "Translate 1 meter"
        - **Combined operations**: "Add window and rotate 90 degrees"

        ### Example Instructions
        - "Add a rectangular window on the right wall"
        - "Rotate the door 90 degrees and open outward"
        - "Add a cylindrical column at the top"
        - "Remove the left window and add a new door"
        """)


def batch_processing_page():
    """Batch processing page"""
    st.markdown('<h2 class="section-header">📦 Batch Processing</h2>', unsafe_allow_html=True)

    # Batch upload
    uploaded_files = st.file_uploader(
        "Batch upload image files",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True,
        help="Select multiple image files for batch processing"
    )

    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} files")

        # Batch text instructions
        st.subheader("📝 Batch Editing Instructions")
        use_same_instruction = st.checkbox("Use same editing instruction for all images", value=True)

        if use_same_instruction:
            batch_instruction = st.text_area(
                "Enter editing instruction",
                placeholder="Example: Add window",
                height=100
            )
            instructions = [batch_instruction] * len(uploaded_files)
        else:
            st.write("Enter editing instruction for each image:")
            instructions = []
            for i, file in enumerate(uploaded_files):
                instruction = st.text_input(
                    f"File {i + 1}: {file.name}",
                    key=f"instruction_{i}",
                    placeholder="Leave blank for no editing"
                )
                instructions.append(instruction)

        # Process button
        if st.button("🚀 Start Batch Processing", type="primary"):
            if not all(instructions):
                st.warning("Please provide editing instructions for all images, or leave blank for no editing")
                return

            # Create progress display
            progress_container = st.container()
            results_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            results = []
            temp_dir = tempfile.mkdtemp()

            try:
                for i, (uploaded_file, instruction) in enumerate(zip(uploaded_files, instructions)):
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {uploaded_file.name} ({i + 1}/{len(uploaded_files)})")

                    # Save file
                    image_path = save_uploaded_file(uploaded_file, temp_dir)

                    if image_path:
                        # Process image
                        result = REASONING_ENGINE.process_single_image(
                            image_path, instruction, f"{args.output_dir}/batch_{i + 1}"
                        )
                        results.append(result)

                        # Display results
                        with results_container:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.image(uploaded_file, width=100)
                            with col2:
                                if result.get('success'):
                                    st.success(f"✅ {uploaded_file.name} - Processing complete")
                                else:
                                    st.error(f"❌ {uploaded_file.name} - Processing failed: {result.get('error')}")

                # Display summary
                success_count = sum(1 for r in results if r.get('success'))
                st.success(f"Batch processing complete! Success: {success_count}/{len(results)}")

            finally:
                # Clean up temporary files
                shutil.rmtree(temp_dir)


def interactive_editing_page():
    """Interactive editing page"""
    st.markdown('<h2 class="section-header">✏️ Interactive Editing</h2>', unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload initial image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload initial image for interactive editing"
    )

    if uploaded_file is not None:
        # Display initial image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📸 Initial Image")
            st.image(uploaded_file, use_column_width=True)

        with col2:
            st.subheader("📝 Edit History")

            # Initialize edit history
            if 'edit_history' not in st.session_state:
                st.session_state.edit_history = []

            # Add new edit
            new_instruction = st.text_input(
                "Enter editing instruction",
                placeholder="Example: Add window",
                key="new_edit"
            )

            col3, col4 = st.columns([1, 1])
            with col3:
                if st.button("➕ Add Edit", use_container_width=True):
                    if new_instruction.strip():
                        st.session_state.edit_history.append(new_instruction.strip())
                        st.rerun()

            with col4:
                if st.button("🔄 Clear History", use_container_width=True):
                    st.session_state.edit_history.clear()
                    st.rerun()

            # Display edit history
            if st.session_state.edit_history:
                st.write("**Current Edit History:**")
                for i, instruction in enumerate(st.session_state.edit_history):
                    st.text(f"{i + 1}. {instruction}")
            else:
                st.info("No edit history yet")

        # Execute edits
        if st.session_state.edit_history and st.button("🚀 Apply All Edits", type="primary"):
            with st.spinner("Applying edits..."):
                # Save file
                temp_dir = tempfile.mkdtemp()
                image_path = save_uploaded_file(uploaded_file, temp_dir)

                if image_path:
                    # Process initial image
                    initial_result = REASONING_ENGINE.process_single_image(image_path, "", temp_dir)

                    if initial_result['success']:
                        # Execute interactive editing
                        edit_result = REASONING_ENGINE.interactive_edit(
                            initial_result['final_mesh'],
                            st.session_state.edit_history
                        )

                        # Display results
                        st.success("✅ Editing complete!")

                        # Display final mesh
                        fig = visualize_3d_mesh(edit_result['final_mesh'])
                        st.plotly_chart(fig, use_container_width=True)

                        # Display edit details
                        with st.expander("📊 Edit Details"):
                            for result in edit_result['edit_results']:
                                if result['success']:
                                    st.success(f"Step {result['step']}: {result['instruction']}")
                                    if 'stability_analysis' in result:
                                        stability = result['stability_analysis']
                                        st.info(f"Stability: {'✅ Stable' if stability.get('is_stable') else '❌ Unstable'}")
                                else:
                                    st.error(f"Step {result['step']}: {result['instruction']} - Failed")

                        # Download final model
                        temp_mesh_path = os.path.join(temp_dir, "final_mesh.ply")
                        o3d.io.write_triangle_mesh(temp_mesh_path, edit_result['final_mesh'])

                        with open(temp_mesh_path, "rb") as f:
                            mesh_bytes = f.read()

                        st.download_button(
                            label="Download Final Edited Result",
                            data=mesh_bytes,
                            file_name="edited_model.ply",
                            mime="application/octet-stream"
                        )

                    # Clean up temporary files
                    shutil.rmtree(temp_dir)


def results_viewer_page():
    """Results viewer page"""
    st.markdown('<h2 class="section-header">📊 Results Viewer</h2>', unsafe_allow_html=True)

    # Select results directory
    results_dir = st.text_input(
        "Results directory path",
        value="./output",
        help="Enter path to directory containing processing results"
    )

    if os.path.exists(results_dir):
        # Scan results directory
        results_list = []
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path):
                mesh_file = os.path.join(item_path, "final_mesh.ply")
                info_file = os.path.join(item_path, "inference_results.json")

                if os.path.exists(mesh_file):
                    results_list.append({
                        'name': item,
                        'path': item_path,
                        'mesh_file': mesh_file,
                        'info_file': info_file if os.path.exists(info_file) else None
                    })

        if results_list:
            st.info(f"Found {len(results_list)} results")

            # Select result to view
            selected_result = st.selectbox(
                "Select result",
                results_list,
                format_func=lambda x: x['name']
            )

            if selected_result:
                # Load mesh
                mesh = o3d.io.read_triangle_mesh(selected_result['mesh_file'])

                if mesh.has_vertices():
                    # Display 3D visualization
                    fig = visualize_3d_mesh(mesh)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display detailed information
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Vertices", len(mesh.vertices))

                    with col2:
                        st.metric("Faces", len(mesh.triangles))

                    with col3:
                        if selected_result['info_file']:
                            try:
                                with open(selected_result['info_file'], 'r', encoding='utf-8') as f:
                                    info = json.load(f)
                                st.metric("Processing Time", f"{info.get('processing_time', 0):.2f}s")
                            except:
                                st.metric("Processing Time", "Unknown")

                    # Download button
                    with open(selected_result['mesh_file'], "rb") as f:
                        mesh_bytes = f.read()

                    st.download_button(
                        label="Download 3D Model",
                        data=mesh_bytes,
                        file_name=f"{selected_result['name']}.ply",
                        mime="application/octet-stream"
                    )
                else:
                    st.error("Failed to load 3D model")
        else:
            st.warning("No processing results found")
    else:
        st.error(f"Directory does not exist: {results_dir}")


def main():
    """Main function"""
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # System information
        st.markdown("### System Information")
        if REASONING_ENGINE is not None:
            st.success("✅ System initialized")
        else:
            st.warning("⚠️ System not initialized")

        # Configuration selection
        config_mode = st.selectbox(
            "Configuration Mode",
            ["Fast Mode", "Standard Mode", "Production Mode"],
            index=0
        )

        # Performance settings
        st.markdown("### Performance Settings")
        use_gpu = st.checkbox("Use GPU", value=torch.cuda.is_available())
        mixed_precision = st.checkbox("Mixed Precision", value=True)

        # About information
        st.markdown("### About LSVR-SE")
        st.markdown("""
        **LSVR-SE** is an advanced 3D reconstruction and editing system,
        supporting generation of editable 3D models from single images.

        **Core Features:**
        - 🧠 Intelligent 3D Reconstruction
        - ✏️ Semantic Editing
        - 🎨 Real-time Rendering
        - 📊 Quality Analysis
        """)

        st.markdown("### Version Information")
        st.text("Version: 1.0.0")
        st.text("Build Date: 2025-12-01")

    # Main content
    main_page()


if __name__ == "__main__":
    main()