@echo off
REM Build CUDA kernels for Motion Transfer nodes (Windows)
REM
REM Requirements:
REM   - CUDA Toolkit 11.x or 12.x
REM   - Visual Studio 2019/2022 with C++ build tools
REM   - nvcc compiler in PATH
REM
REM Usage:
REM   build.bat
REM

echo Building Motion Transfer CUDA kernels...

REM Detect CUDA architecture (adjust for your GPU)
REM Common values:
REM   - RTX 30xx/40xx: compute_86,sm_86 or compute_89,sm_89
REM   - RTX 20xx: compute_75,sm_75
REM   - GTX 10xx: compute_61,sm_61

REM Use nvidia-smi to detect GPU
nvidia-smi --query-gpu=name --format=csv,noheader > gpu.tmp
set /p GPU_NAME=<gpu.tmp
del gpu.tmp

echo Detected GPU: %GPU_NAME%

REM Map GPU to compute capability (heuristic)
set SM_ARCH=compute_75,sm_75
if "%GPU_NAME:RTX 40=%" neq "%GPU_NAME%" set SM_ARCH=compute_89,sm_89
if "%GPU_NAME:RTX 30=%" neq "%GPU_NAME%" set SM_ARCH=compute_86,sm_86
if "%GPU_NAME:RTX 20=%" neq "%GPU_NAME%" set SM_ARCH=compute_75,sm_75
if "%GPU_NAME:GTX 10=%" neq "%GPU_NAME%" set SM_ARCH=compute_61,sm_61

echo Compiling for architecture: %SM_ARCH%
echo.

REM Compile TileWarp kernel
echo [1/3] Compiling tile_warp_kernel.cu...
nvcc -shared ^
    -arch=%SM_ARCH% ^
    -O3 ^
    -o tile_warp_kernel.dll ^
    tile_warp_kernel.cu

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to compile tile_warp_kernel.cu
    exit /b 1
)

REM Compile BarycentricWarp kernel
echo [2/3] Compiling barycentric_warp_kernel.cu...
nvcc -shared ^
    -arch=%SM_ARCH% ^
    -O3 ^
    -o barycentric_warp_kernel.dll ^
    barycentric_warp_kernel.cu

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to compile barycentric_warp_kernel.cu
    exit /b 1
)

REM Compile GuidedFilter kernel
echo [3/3] Compiling guided_filter_kernel.cu...
nvcc -shared ^
    -arch=%SM_ARCH% ^
    -O3 ^
    -o guided_filter_kernel.dll ^
    guided_filter_kernel.cu

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to compile guided_filter_kernel.cu
    exit /b 1
)

echo.
echo ✓ Build complete! CUDA kernels compiled successfully.
echo.
echo Generated libraries:
dir /b *.dll
echo.
echo You can now use CUDA acceleration in ComfyUI Motion Transfer nodes.
pause
