@echo off
title nncase Compiler Compilation & Quantization Runner
echo =======================================================================
echo nncase Compiler (ncc) Runner for K210 YOLOv3-Lite Model
echo =======================================================================
echo.

echo [1/2] Preparing calibration dataset...
python convert/prepare_calibration.py
echo.

echo [2/2] Running nncase compiler...
echo Checking for 'ncc' executable...
where ncc >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: 'ncc' was not found in your system PATH.
    echo Please download the nncase compiler v0.1.0-rc5 from:
    echo https://github.com/kendryte/nncase/releases/tag/v0.1.0-rc5
    echo Extract ncc.exe and ensure it is either in your PATH or placed in this folder.
    echo.
    set /p NCC_PATH="Enter the absolute path to ncc.exe (or press Enter to exit): "
    if "%NCC_PATH%"=="" exit /b 1
) else (
    set NCC_PATH=ncc
)

echo.
echo Running: "%NCC_PATH%" compile convert/yolov3_lite.tflite deploy/yolov3_lite.kmodel -i tflite -o kmodel --dataset convert/calibration_dataset --inference-type uint8 --input-type float --dataset-format image --input-layout NCHW --mean 0 --std 1
echo.

"%NCC_PATH%" compile convert/yolov3_lite.tflite deploy/yolov3_lite.kmodel -i tflite -o kmodel --dataset convert/calibration_dataset --inference-type uint8 --input-type float --dataset-format image --input-layout NCHW --mean 0 --std 1

if %errorlevel% neq 0 (
    echo.
    echo =======================================================================
    echo ERROR: nncase compilation failed!
    echo Common reasons:
    echo 1. The input tflite path is incorrect.
    echo 2. The input tflite has operators that are unsupported by the KPU compiler.
    echo 3. The dataset format or input dimensions do not match.
    echo =======================================================================
    pause
    exit /b 1
)

echo.
echo =======================================================================
echo SUCCESS: Model compiled successfully!
echo Output saved to: deploy/yolov3_lite.kmodel
echo =======================================================================
pause
