
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
def Landing():
    return render_template('Home.html')

@app.route('/Home.html')
def Home():
    return render_template('Home.html')

@app.route('/Search.html')
def Search():
    return render_template('Search.html')

@app.route('/About.html')
def About():
    return render_template('About.html')

def plotgraph():
    end = datetime.now()
    start = datetime(end.year-1,end.month,end.day)
    stock_df = get_history(symbol="SBIN",start=start,end=end)
    fig = plt.figure(figsize=(10,5))
    plt.xlabel('Year-Month') 
    plt.ylabel('Stock-Price(in Rupees.)') 
    plt.title('Closing price of the stock for last 1 Year') 
    stock_df['Close'].plot(legend=True, figsize=(10,4))
    fig.savefig('static/stock-plot.jpg', bbox_inches='tight', dpi=150)

def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''
    dt = 1/days
    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run price array for number of days
    for x in range(1,days):
        
        # Calculate Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price


@app.route('/stock_pred/',methods = ['POST'])
def stock_pred():
    # user_input = request.form['fn']
    try:
        plotgraph()
        end = datetime.now()
        start = datetime(end.year-1,end.month,end.day)
        b = []
        stock_df = get_history(symbol="SBIN",start=start,end=end)
        closingprice_df = stock_df['Close']
        closingprice_df=closingprice_df.to_frame()
        tech_returns = closingprice_df.pct_change()
        rets = tech_returns.dropna()
        days = 365
        mu = rets.mean()['Close']
        sigma = rets.std()['Close']
        start_price = closingprice_df.iloc[-1, closingprice_df.columns.get_loc("Close")]
        runs = 10000
        simulations = np.zeros(runs)
        for run in range(runs):    
            simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
        max_low = np.percentile(simulations,1)
        num_stocks=10000/start_price
        current_value=round(simulations.mean()*num_stocks,2)
        profit_percent=round(((current_value-10000)/10000)*100)
        max_loss=round((start_price-max_low)*num_stocks,2)
        loss_percent=round((max_loss/10000)*100,2)
        b.append(current_value)
        b.append(profit_percent)
        b.append(max_loss)
        b.append(loss_percent)
        print(b)
        return  render_template('Result.html',predictions={"current_value": current_value, "profit_percent":profit_percent,"max_loss":max_loss,"loss_percent":loss_percent})
    
    

    except:
        return render_template('error.html')


# Any HTML template in Flask App render_template

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of your flask app or api


