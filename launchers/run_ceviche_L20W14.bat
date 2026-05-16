@echo off
chcp 65001 > nul
cd /d C:\Users\Hemera\Projects\meep_1x8_progress
set PY=C:\Users\Hemera\anaconda3\envs\photon\python.exe
set ROOT=C:\Users\Hemera\Projects\meep_1x8_progress
set SCRIPTS=%ROOT%\scripts
set S1LOG=%ROOT%\ceviche_L20W14_s1.log
set S2LOG=%ROOT%\ceviche_L20W14_s2.log
set EVLOG=%ROOT%\ceviche_L20W14_eval.log

echo === Ceviche S1 L20W14 === > "%S1LOG%"
echo Start: %date% %time% >> "%S1LOG%"
"%PY%" "%SCRIPTS%\ceviche_stage1.py" --mmi_L 20 --mmi_W 14 --n_iters 100 >> "%S1LOG%" 2>&1
echo Done: %date% %time% >> "%S1LOG%"

echo === Ceviche S2 L20W14 === > "%S2LOG%"
echo Start: %date% %time% >> "%S2LOG%"
"%PY%" "%SCRIPTS%\ceviche_stage2.py" --mmi_L 20 --mmi_W 14 --n_iters 60 >> "%S2LOG%" 2>&1
echo Done: %date% %time% >> "%S2LOG%"

echo === Ceviche Eval L20W14 === > "%EVLOG%"
"%PY%" "%SCRIPTS%\ceviche_eval.py" --dir L20W14_s2 >> "%EVLOG%" 2>&1
echo Eval done: %date% %time% >> "%EVLOG%"
