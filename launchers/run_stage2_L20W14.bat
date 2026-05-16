@echo off
cd /d C:\Users\Hemera\Projects\meep_1x8_progress
set S2LOG=C:\Users\Hemera\Projects\meep_1x8_progress\ceviche\L20W14_s2_run.log
echo Stage2 start: %date% %time% > "%S2LOG%"
call conda activate photon
python scripts\ceviche_stage2.py --mmi_L 20 --mmi_W 14 --n_iters 60 >> "%S2LOG%" 2>&1
echo Stage2 done: %date% %time% >> "%S2LOG%"
