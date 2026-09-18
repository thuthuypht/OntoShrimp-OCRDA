@echo off
REM ============================================================
REM EDIT THESE 4 PATHS BEFORE RUNNING
REM ============================================================

set D1=D:\ShrimpDataset
set D2=D:\TigerShrimpBD
set D3=D:\Dataset3_Vietnam_v1.zip
set OUT=D:\DS3_cross_dedup_results

python check_cross_dataset_duplicates.py --d1 "%D1%" --d2 "%D2%" --d3 "%D3%" --out "%OUT%"

pause
