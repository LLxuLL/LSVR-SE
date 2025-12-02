#!/bin/bash
# LSVR-SE Linux/Mac Launcher Script
# Supports multiple launch modes: Web application, training, inference

set -e  # Exit immediately on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function: Print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}        LSVR-SE Launcher (Linux/Mac)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  Warning: $1${NC}"
}

# Check Python environment
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 environment not found"
        print_info "Please install Python 3.8-3.10"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $PYTHON_VERSION"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    # Check Streamlit
    if ! python3 -c "import streamlit" &> /dev/null; then
        print_error "Streamlit library not found"
        print_info "Please run: pip install streamlit"
        exit 1
    fi

    # Check PyTorch
    if ! python3 -c "import torch" &> /dev/null; then
        print_error "PyTorch library not found"
        print_info "Please run: pip install torch torchvision torchaudio"
        exit 1
    fi

    # Check CUDA availability
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_success "CUDA available"
    else
        print_warning "CUDA not available, will use CPU mode"
    fi
}

# Launch Web application
start_webapp() {
    print_info "Launching Web application..."
    print_info "Application will open in browser"
    print_info "Default address: http://localhost:8501"
    echo

    # Check application file
    if [ ! -f "application.py" ]; then
        print_error "application.py file not found"
        print_info "Please make sure you are running this script from the correct directory"
        exit 1
    fi

    print_success "Starting Streamlit application..."
    streamlit run application.py --server.port=8501 --server.address=localhost
}

# Launch training
start_training() {
    print_info "Launching Training Mode..."
    print_info "Please ensure training data is prepared"
    echo

    # Check training script
    if [ ! -f "train.py" ]; then
        print_error "train.py file not found"
        exit 1
    fi

    # Training options
    echo "Select training configuration:"
    echo "1. Fast training (for testing)"
    echo "2. Standard training (recommended)"
    echo "3. Production training (full training)"
    echo

    read -p "Enter training mode (1-3): " train_mode

    case $train_mode in
        1)
            config="--config fast"
            ;;
        2)
            config="--config default"
            ;;
        3)
            config="--config production"
            ;;
        *)
            print_error "Invalid training mode, using default configuration"
            config="--config default"
            ;;
    esac

    echo
    print_success "Starting training..."
    python3 train.py $config --use_wandb --num_epochs 100
}

# Launch inference
start_inference() {
    print_info "Launching Inference Mode..."
    print_info "Supports single image processing, batch processing, and interactive editing"
    echo

    # Check inference script
    if [ ! -f "reasoning.py" ]; then
        print_error "reasoning.py file not found"
        exit 1
    fi

    # Inference options
    echo "Select inference mode:"
    echo "1. Single image processing"
    echo "2. Batch processing"
    echo "3. Interactive editing"
    echo

    read -p "Enter inference mode (1-3): " inference_mode

    case $inference_mode in
        1)
            echo
            print_info "📸 Single image processing mode"
            read -p "Enter image path: " image_path
            read -p "Enter edit instruction (optional): " text_instruction

            if [ -z "$image_path" ]; then
                print_error "Image path cannot be empty"
                exit 1
            fi

            echo
            print_success "Processing image..."
            python3 reasoning.py --mode single --image "$image_path" --text "$text_instruction"
            ;;
        2)
            echo
            print_info "📦 Batch processing mode"
            read -p "Enter image list file path: " image_list

            if [ -z "$image_list" ]; then
                print_error "Image list file path cannot be empty"
                exit 1
            fi

            echo
            print_success "Starting batch processing..."
            python3 reasoning.py --mode batch --image_list "$image_list"
            ;;
        3)
            echo
            print_info "✏️ Interactive editing mode"
            read -p "Enter initial image path: " image_path

            if [ -z "$image_path" ]; then
                print_error "Image path cannot be empty"
                exit 1
            fi

            echo
            print_success "Starting interactive editing..."
            python3 reasoning.py --mode interactive --image "$image_path"
            ;;
        *)
            print_error "Invalid inference mode"
            exit 1
            ;;
    esac
}

# Launch TensorBoard
start_tensorboard() {
    print_info "Launching TensorBoard..."
    print_info "TensorBoard will open at http://localhost:6006"
    echo

    print_success "Starting TensorBoard..."
    python3 -m tensorboard.main --logdir=./logs --host=localhost --port=6006
}

# Show help
show_help() {
    echo
    echo "📚 LSVR-SE Usage Help"
    echo
    echo "Launch mode descriptions:"
    echo "  1. Web Application     - Launch Streamlit web interface for interactive operations"
    echo "  2. Training Mode       - Start model training, requires prepared training data"
    echo "  3. Inference Mode      - Start inference service, supports multiple processing modes"
    echo "  4. TensorBoard         - Launch visualization tool to view training progress"
    echo "  5. Help Information    - Display this help information"
    echo "  6. Exit                - Exit the launcher"
    echo
    echo "Environment requirements:"
    echo "  - Python 3.8-3.10"
    echo "  - PyTorch 1.12+ (CUDA 11.8+)"
    echo "  - 16GB+ GPU memory (recommended)"
    echo "  - 64GB+ system memory (recommended)"
    echo
    echo "File structure:"
    echo "  - src/          Source code directory"
    echo "  - models/       Model files directory"
    echo "  - data/         Dataset directory"
    echo "  - output/       Output results directory"
    echo "  - checkpoints/  Training checkpoints directory"
    echo "  - logs/         Log files directory"
    echo
    echo "For more information, please check the documentation in the docs/ directory"
    echo
    read -p "Press Enter to continue..."
}

# Main function
main() {
    print_header

    # Check environment
    check_python
    check_dependencies

    # Display launch options
    echo "Please select launch mode:"
    echo "1. 🌐 Launch Web Application (Recommended)"
    echo "2. 🚀 Launch Training Mode"
    echo "3. 🔍 Launch Inference Mode"
    echo "4. 📊 Launch TensorBoard"
    echo "5. ❓ Show Help Information"
    echo "6. 🚪 Exit"
    echo

    read -p "Enter option number (1-6): " mode

    case $mode in
        1)
            start_webapp
            ;;
        2)
            start_training
            ;;
        3)
            start_inference
            ;;
        4)
            start_tensorboard
            ;;
        5)
            show_help
            ;;
        6)
            echo
            echo "========================================"
            echo "        LSVR-SE Launcher Exited"
            echo "========================================"
            echo
            exit 0
            ;;
        *)
            print_error "Invalid option, please run the script again"
            exit 1
            ;;
    esac
}

# Check if running in interactive mode
if [ -t 0 ]; then
    # Interactive mode
    main "$@"
else
    # Non-interactive mode, show help
    print_header
    echo "LSVR-SE Launcher Script"
    echo "Usage: ./run.sh"
    echo "Or use command line arguments to specify mode:"
    echo "  ./run.sh webapp     - Launch Web application"
    echo "  ./run.sh train      - Launch training mode"
    echo "  ./run.sh inference  - Launch inference mode"
    echo "  ./run.sh tensorboard - Launch TensorBoard"
    echo "  ./run.sh help       - Show help information"
fi