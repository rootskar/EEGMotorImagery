#!/usr/bin/env bash

python3.7 -m venv env &&
source env/bin/activate && python3.7 -m pip install --upgrade pip && python3.7 -m pip install wheel && python3.7 -m pip
install -r requirements.txt && python3.7 -m pip install pyedflib==0.1.17