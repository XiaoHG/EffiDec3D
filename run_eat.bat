@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set "PYTHONPATH="
set "CUDA_VISIBLE_DEVICES=0"

:: ==============================================
:: EffiDec3D Windows 训练脚本（PowerShell兼容版）
:: 用法：
::   run_eat.bat smoke     快速测试
::   run_eat.bat train     正式训练
::   run_eat.bat test      验证模型
:: ==============================================

set "PROJ_DIR=D:\EffiDec3D_EAT\EffiDec3D"
set "DATA_ROOT=D:\EffiDec3D_EAT\EffiDec3D_work"
set "OUTPUT_DIR=%PROJ_DIR%\output_folder\eat_run1"
set "CONDA_ENV=effidec3d"

set "DATASET=EAT"
set "NETWORK=3DUXNET_EffiDec3D"
set "IMG_SIZE=96 96 96"
set "N_CHANNELS=1"
set "CHANNELS=48 96 192 384"
set "N_DEC=48"

set "MODE_ARG=%~1"
if not defined MODE_ARG set "MODE_ARG=smoke"

if "%MODE_ARG%"=="smoke" (
    set MODE=train
    set MAX_ITER=40
    set EVAL_STEP=20
    set BATCH=1
    set CROP=1
    set LR=0.001
    set CACHE=0.0
    set WORKERS=0
    set OVERLAP=0.5
) else if "%MODE_ARG%"=="train" (
    set MODE=train
    set MAX_ITER=2000
    set EVAL_STEP=200
    set BATCH=1
    set CROP=2
    set LR=0.001
    set CACHE=0.0
    set WORKERS=0
    set OVERLAP=0.5
) else if "%MODE_ARG%"=="test" (
    set MODE=validation
    set MAX_ITER=1
    set EVAL_STEP=1
    set BATCH=1
    set CROP=1
    set LR=0.001
    set CACHE=0.0
    set WORKERS=0
    set OVERLAP=0.5
) else (
    echo 错误：不支持的模式 %MODE_ARG%
    pause
    exit /b 1
)

cd /d "%PROJ_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ======================================================
echo  训练模式：%MODE_ARG%
echo  数据集：%DATA_ROOT%
echo ======================================================
echo.

:: ========== 强制激活环境，确保一定成功 ==========
call D:\conda_envs\%CONDA_ENV%\Scripts\activate.bat

:: ========== 强制打印信息 ==========
echo [INFO] 当前Python：
python --version
echo [INFO] 开始运行...
echo.

:: ========== 运行训练脚本 ==========
python main_train_BTCV_TU.py ^
--root "%DATA_ROOT%" ^
--output "%OUTPUT_DIR%" ^
--dataset "%DATASET%" ^
--img_size %IMG_SIZE% ^
--n_channels %N_CHANNELS% ^
--network "%NETWORK%" ^
--channels %CHANNELS% ^
--n_decoder_channels %N_DEC% ^
--ds False ^
--mode "%MODE%" ^
--pretrain False ^
--batch_size %BATCH% ^
--crop_sample %CROP% ^
--lr %LR% ^
--optim AdamW ^
--max_iter %MAX_ITER% ^
--eval_step %EVAL_STEP% ^
--val_batch 1 ^
--gpu 0 ^
--cache_rate %CACHE% ^
--num_workers %WORKERS% ^
--overlap %OVERLAP% ^
--skip_aggregation addition ^
--resolution_factor 2

if %errorlevel% neq 0 (
    echo.
    echo ======================================================
    echo  训练失败！错误代码：%errorlevel%
    echo ======================================================
    pause
    exit /b 1
)

echo.
echo ======================================================
echo  运行成功！
echo ======================================================
echo.
pause