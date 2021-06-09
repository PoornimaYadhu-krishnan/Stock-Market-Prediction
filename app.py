
from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from nsepy import get_history
import pyfolio as pf
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)  # intitialize the flaks app  # common 

@app.route('/')
def Home():
    return render_template('Home.html')


@app.route('/stock_pred/',methods = ['POST'])
def stock_pred():
    user_input = request.form['fn']

    try:
        return  render_template('Result.html',predictions=result_list)
    

    except:
        return render_template('error.html')


# Any HTML template in Flask App render_template

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of your flask app or api


