import os
import trimesh
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def check_mesh_quality(root_dir):
    """Check mesh quality without modification"""
    good_files = 0
    bad_files = []

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.endswith('.obj') or fname.startswith('original_'):
                continue

            obj_path = os.path.join(class_dir, fname)
            try:
                mesh = trimesh.load(obj_path)

                # Basic quality check
                if mesh.is_empty:
                    raise ValueError("Empty mesh")
                if len(mesh.faces) < 4:
                    raise ValueError(f"Insufficient faces: {len(mesh.faces)}")
                if not mesh.is_watertight:
                    raise ValueError("Non-watertight mesh")

                good_files += 1

            except Exception as e:
                bad_files.append((obj_path, str(e)))
                logging.warning(f"Problem file: {obj_path} - {str(e)}")

    logging.info(f"Check completed: Good {good_files}, Problems {len(bad_files)}")

    # Save list of problematic files
    if bad_files:
        with open(os.path.join(root_dir, "problem_files.txt"), 'w') as f:
            for path, reason in bad_files:
                f.write(f"{path} - {reason}\n")

    return bad_files


if __name__ == "__main__":
    dataset_path = "../data/shape_prior_dataset/full"
    bad_files = check_mesh_quality(dataset_path)
    print(f"Found {len(bad_files)} problematic files")