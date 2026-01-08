@echo off
setlocal enabledelayedexpansion

set bit_list=16 32 64 128
set noise_ratio_list=0.2 0.4 0.6 0.8
set dataset=xmedia

for %%b in (%bit_list%) do (
    for %%n in (%noise_ratio_list%) do (
        echo =====================================
        echo Running experiment with:
        echo Bit: %%b, Noise Ratio: %%n
        echo =====================================

        python train.py ^
            --dataset !dataset! ^
            --noisy_ratio %%n ^
            --bit %%b
    )
)

echo All experiments completed!
endlocal