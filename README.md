# D.D.I detective backend API.

This repository contains the code for D.D.I. detective, a web application that utilizes machine learning models(XGBoost), to predict the interaction between two drugs, given their SMILES strings as input

## Installation

- clone the repo `https://github.com/valentina-buoro/Drug-Interaction-Backend.git`
- go into the folder `cd Drug-Interaction-Backend`
- create a virtual python environment, mac `python3 -m venv .venv`. Windows `py -3 -m venv .venv`
- activate the virtual python environment. Mac `. .venv/bin/activate`. Windows `.venv\Scripts\activate`
- install the dependencies `pip install -r requirements.txt`
- run the application `flask --app flaskr run --debug`