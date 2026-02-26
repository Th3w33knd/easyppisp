@echo off
echo Initializing MSVC environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo Failed to initialize MSVC environment.
    exit /b %errorlevel%
)

echo Setting up environment variables...
set "PYTHONPATH=src"
set "CUDA_HOME=C:\cuda_env"
set "LIB=%LIB%;C:\cuda_env\lib;C:\cuda_env\lib\x64"
set "INCLUDE=%INCLUDE%;C:\cuda_env\include;C:\cuda_env\include\targets\x64"
set "PATH=C:\cuda_env\bin;%PATH%"
set DISTUTILS_USE_SDK=1

echo CUDA_HOME=%CUDA_HOME%
echo.
echo Building CUDA extension...
.pixi\envs\gpu\python.exe setup_local.py build_ext --inplace
if %errorlevel% neq 0 (
    echo Build failed.
    exit /b %errorlevel%
)
echo Build successful!
