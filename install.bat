@ECHO OFF
ECHO Creating new virtual environment
python -m venv venv
ECHO Activating virtual environment
call venv\Scripts\activate.bat
ECHO Installing requirements
pip install -r requirements.txt
pip install pyedflib==0.1.17