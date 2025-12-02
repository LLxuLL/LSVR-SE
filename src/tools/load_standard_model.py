import os
import shutil
import argparse


def load_standard_model(model_path, output_dir="./models"):
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "standard_tank_model.obj")

    # Check if source and target files are the same
    if os.path.abspath(model_path) == os.path.abspath(target_path):
        print(f"Source and target files are the same, no need to copy. Standard tank model already at: {target_path}")
        return True

    try:
        # Copy model to target location
        shutil.copy(model_path, target_path)
        print(f"Standard tank model successfully loaded to: {target_path}")
        return True
    except Exception as e:
        print(f"Failed to load standard tank model: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load standard tank model')
    parser.add_argument('--model', type=str, required=True, help='Standard tank model path')
    parser.add_argument('--output', type=str, default="./models", help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist at {args.model}")
    else:
        success = load_standard_model(args.model, args.output)
        if success:
            print("Standard tank model loaded successfully!")
        else:
            print("Standard tank model loading failed")