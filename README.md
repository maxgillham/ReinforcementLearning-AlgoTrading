# Reinforcement Learning for Algorithmic Trading

## Getting Started
**Python 3.6.5**    

I would reccomend creating a virutal enviorment named `env` inside the repository so its included already in the .gitignore
You can create a virtual enviorment using [Virtualenv]("https://virtualenv.pypa.io/en/latest/") if you don't already have it installed in your current python enviorment.  The current dependancies are in `requirements-cpu.txt` and can be installed by the following commands.  
```
pip install virtualenv
virtualenv env

//For MacOs or Linux users
source activate env

//For Windows
cd ./env/Scripts
activate.bat

pip install -r requirements-cpu.txt
```  
The equivalent requirements for gpu support are inside `requirements-gpu.txt` but this isn't necesary unless tensorflow is used.  
To make sure you are configured, run `python main.py`.  

