# alpaca-predict

This is an exploration into Agentic AI in the stock market. By using a Hidden Markov Model, I am predicting the next days state to buy/sell and set take-profit and stop-loss margins on positions. This current version is Agentic as it is easily configurable to set capital limits, strategies and other configuration parameters. 


## How to try it yourself:

1. Clone the repo
2. Create a virtual environment
    - Windows <code>python -m virtualenv .venv</code>
    - Mac/Linux <code>python -m virtualenv .venv</code>
3. Activate virtual environment
    - Windows <code>.venv\Scripts\activate</code>
    - Mac/Linux <code>source .venv/bin/activate</code>
4. Install packages
    - Windows <code>pip install -r requirements.txt</code>
    - Mac/Linux <code>pip install -r requirements.txt</code>
5. Run trader.py
    - Windows <code>python trader.py</code>
    - Mac/Linux <code>python trader.py</code>

### Note
This program was designed to use websockets and run constantly, monitoring the changes in an asset during market hours. I accomplished this on Linux by using a systemctl service, if you're on a Windows server, good luck. 