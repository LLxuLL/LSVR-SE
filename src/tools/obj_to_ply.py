import os
import argparse
import numpy as np
import open3d as o3d


def convert_obj_to_ply(obj_path, ply_path):
    """
    Convert OBJ file to point cloud PLY file
    :param obj_path: Input OBJ file path
    :param ply_path: Output PLY file path
    """
    # Read OBJ file
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # Extract vertices as point cloud
    vertices = np.asarray(mesh.vertices)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Save as PLY file
    o3d.io.write_point_cloud(ply_path, pcd)
    return len(vertices)


def process_class(class_name, base_input_dir, base_output_dir):
    """
    Process all OBJ files for a specific class
    :param class_name: Class name (e.g., 'apple')
    :param base_input_dir: Base directory for input OBJ files
    :param base_output_dir: Base directory for output PLY files
    """
    input_dir = os.path.join(base_input_dir, class_name)
    output_dir = os.path.join(base_output_dir, class_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Count converted files
    converted_count = 0
    total_points = 0

    print(f"\nProcessing class: {class_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Iterate through all OBJ files
    for filename in os.listdir(input_dir):
        if filename.endswith('.obj'):
            obj_path = os.path.join(input_dir, filename)
            ply_filename = os.path.splitext(filename)[0] + '.ply'
            ply_path = os.path.join(output_dir, ply_filename)

            # Convert file
            try:
                num_points = convert_obj_to_ply(obj_path, ply_path)
                converted_count += 1
                total_points += num_points
                print(f"Converted: {filename} -> {ply_filename} ({num_points} points)")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

    return converted_count, total_points


def main():
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Convert OBJ files to point cloud PLY format.')
    parser.add_argument('--class_name', type=str, default='all',
                        help='Specific class name to convert (default: all classes)')
    parser.add_argument('--input_base', type=str, default='../data/shape_prior_dataset/full',
                        help='Base directory for input OBJ files')
    parser.add_argument('--output_base', type=str, default='../data/shape_prior_dataset/point_cloud',
                        help='Base directory for output PLY files')

    args = parser.parse_args()

    # Verify directory exists
    if not os.path.exists(args.input_base):
        print(f"Error: Input directory not found: {args.input_base}")
        return

    # Create output base directory
    os.makedirs(args.output_base, exist_ok=True)

    # Determine classes to process
    if args.class_name.lower() == 'all':
        # Process all classes
        class_names = [d for d in os.listdir(args.input_base)
                       if os.path.isdir(os.path.join(args.input_base, d))]
        print(f"Found {len(class_names)} classes to process")
    else:
        # Process specified class
        class_names = [args.class_name]
        if not os.path.exists(os.path.join(args.input_base, args.class_name)):
            print(f"Error: Class directory not found: {os.path.join(args.input_base, args.class_name)}")
            return

    # Process each class
    total_converted = 0
    total_points = 0

    for class_name in class_names:
        converted, points = process_class(class_name, args.input_base, args.output_base)
        total_converted += converted
        total_points += points

    print("\nConversion Summary:")
    print(f"Total classes processed: {len(class_names)}")
    print(f"Total OBJ files converted: {total_converted}")
    print(f"Total points processed: {total_points}")
    print(f"Output directory: {args.output_base}")


if __name__ == "__main__":
    main()