@echo off
python -m pip install -r requirements.txt
python src\train.py --quick
pause
