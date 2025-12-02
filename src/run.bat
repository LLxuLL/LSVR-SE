@echo off
REM LSVR-SE Windows Launcher Script
REM Supports multiple launch modes: Web application, training, inference

echo.
echo ========================================
echo        LSVR-SE Launcher (Windows)
echo ========================================
echo.

REM Check Python environment
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: Python environment not found
    echo    Please ensure Python 3.8-3.10 is installed
    echo    and added to system PATH
    pause
    exit /b 1
)

REM Check Streamlit
python -c "import streamlit" >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Error: Streamlit library not found
    echo    Please run: pip install streamlit
    pause
    exit /b 1
)

REM Display launch options
echo Please select launch mode:
echo.
echo 1. 🌐 Launch Web Application (Recommended)
echo 2. 🚀 Launch Training Mode
echo 3. 🔍 Launch Inference Mode
echo 4. 📊 Launch TensorBoard
echo 5. ❓ Show Help Information
echo 6. 🚪 Exit
echo.

set /p mode=Enter option number (1-6):

if "%mode%"=="1" goto :webapp
if "%mode%"=="2" goto :training
if "%mode%"=="3" goto :inference
if "%mode%"=="4" goto :tensorboard
if "%mode%"=="5" goto :help
if "%mode%"=="6" goto :exit

echo ❌ Invalid option, please run the script again
pause
exit /b 1

:webapp
echo.
echo 🌐 Launching Web Application...
echo    Application will open in browser
echo    Default address: http://localhost:8501
echo.

REM Check application file
if not exist "application.py" (
    echo ❌ Error: application.py file not found
    echo    Please make sure you are running this script from the correct directory
    pause
    exit /b 1
)

echo ✅ Starting Streamlit application...
streamlit run application.py --server.port=8501 --server.address=localhost

goto :exit

:training
echo.
echo 🚀 Launching Training Mode...
echo    Please ensure training data is prepared
echo.

REM Check training script
if not exist "train.py" (
    echo ❌ Error: train.py file not found
    pause
    exit /b 1
)

REM Display training options
echo Select training configuration:
echo 1. Fast training (for testing)
echo 2. Standard training (recommended)
echo 3. Production training (full training)
echo.

set /p train_mode=Enter training mode (1-3):

if "%train_mode%"=="1" (
    set config=--config fast
) else if "%train_mode%"=="2" (
    set config=--config default
) else if "%train_mode%"=="3" (
    set config=--config production
) else (
    echo ❌ Invalid training mode, using default configuration
    set config=--config default
)

echo.
echo ✅ Starting training...
python train.py %config% --use_wandb --num_epochs 100

goto :exit

:inference
echo.
echo 🔍 Launching Inference Mode...
echo    Supports single image processing, batch processing, and interactive editing
echo.

REM Check inference script
if not exist "reasoning.py" (
    echo ❌ Error: reasoning.py file not found
    pause
    exit /b 1
)

REM Display inference options
echo Select inference mode:
echo 1. Single image processing
echo 2. Batch processing
echo 3. Interactive editing
echo.

set /p inference_mode=Enter inference mode (1-3):

if "%inference_mode%"=="1" (
    echo.
    echo 📸 Single image processing mode
    set /p image_path=Enter image path:
    set /p text_instruction=Enter edit instruction (optional):

    if "%image_path%"=="" (
        echo ❌ Image path cannot be empty
        pause
        exit /b 1
    )

    echo.
    echo ✅ Processing image...
    python reasoning.py --mode single --image "%image_path%" --text "%text_instruction%"

) else if "%inference_mode%"=="2" (
    echo.
    echo 📦 Batch processing mode
    set /p image_list=Enter image list file path:

    if "%image_list%"=="" (
        echo ❌ Image list file path cannot be empty
        pause
        exit /b 1
    )

    echo.
    echo ✅ Starting batch processing...
    python reasoning.py --mode batch --image_list "%image_list%"

) else if "%inference_mode%"=="3" (
    echo.
    echo ✏️ Interactive editing mode
    set /p image_path=Enter initial image path:

    if "%image_path%"=="" (
        echo ❌ Image path cannot be empty
        pause
        exit /b 1
    )

    echo.
    echo ✅ Starting interactive editing...
    python reasoning.py --mode interactive --image "%image_path%"

) else (
    echo ❌ Invalid inference mode
    pause
    exit /b 1
)

goto :exit

:tensorboard
echo.
echo 📊 Launching TensorBoard...
echo    TensorBoard will open at http://localhost:6006
echo.

echo ✅ Starting TensorBoard...
python -m tensorboard.main --logdir=./logs --host=localhost --port=6006

goto :exit

:help
echo.
echo 📚 LSVR-SE Usage Help
echo.
echo Launch mode descriptions:
echo   1. Web Application     - Launch Streamlit web interface for interactive operations
echo   2. Training Mode       - Start model training, requires prepared training data
echo   3. Inference Mode      - Start inference service, supports multiple processing modes
echo   4. TensorBoard         - Launch visualization tool to view training progress
echo   5. Help Information    - Display this help information
echo   6. Exit                - Exit the launcher
echo.
echo Environment requirements:
echo   - Python 3.8-3.10
echo   - PyTorch 1.12+ (CUDA 11.8+)
echo   - 16GB+ GPU memory (recommended)
echo   - 64GB+ system memory (recommended)
echo.
echo File structure:
echo   - src/          Source code directory
echo   - models/       Model files directory
echo   - data/         Dataset directory
echo   - output/       Output results directory
echo   - checkpoints/  Training checkpoints directory
echo   - logs/         Log files directory
echo.
echo For more information, please check the documentation in the docs/ directory
echo.
pause
goto :exit

:exit
echo.
echo ========================================
echo        LSVR-SE Launcher Exited
echo ========================================
echo.
pause