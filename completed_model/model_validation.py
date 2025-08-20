import os
import json


def validate_sd_model(model_path):
    """Validate Stable Diffusion model integrity"""
    required_dirs = [
        "scheduler",
        "text_encoder",
        "tokenizer",
        "unet",
        "vae",
        "feature_extractor"
    ]
    required_files = [
        "model_index.json",
        "unet/config.json",
        "vae/config.json"
    ]
    valid = True
    print("Validating Stable Diffusion model structure: ")
    print(f"Model path: {model_path}")

    # Check for required directories
    for subdir in required_dirs:
        path = os.path.join(model_path, subdir)
    if os.path.exists(path):
        print(f" ✓ {subdir}/")
    else:
        print(f" ✗ {subdir}/ (MISSING)")
    valid = False

    # Check for required files
    for file in required_files:
        path = os.path.join(model_path, file)
    if os.path.exists(path):
        print(f" ✓ {file}")
    else:
        print(f" ✗ {file} (MISSING)")
    valid = False

    # Check model_index.json content
    try:
        with open(os.path.join(model_path, "model_index.json"), "r") as f:
            model_index = json.load(f)
        if "diffusers_version" not in model_index:
            print(" ✗ model_index.json missing 'diffusers_version' key")
            valid = False
    except Exception as e:
        print(f" ✗ Failed to read model_index.json: {str(e)}")
        valid = False

    print(f"Model validation: {'SUCCESS' if valid else 'FAILED'}")
    return valid