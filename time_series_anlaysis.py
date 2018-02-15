# Task 1
import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Autumn 2017\\Advance Python\\streaming.csv')

df.dropna()
import matplotlib.pyplot as plt


# Task 2
# simple regession without rolling for red
import statsmodels.api as sm


def calc_stats_red(df):
    X = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(df.iloc[:, 2], X).fit()

    beta = model.params["Alpha"]
    alpha = model.params["const"]
    r2 = model.rsquared
    return beta, alpha, r2

def rolling_stats_r(df, window=501):
    #     dataframe to hold the results
    res = pd.DataFrame(index=df.index)

    for i in range(0,len(df.index)):

        if len(df) - i >= window:
            # break the df into smaller chunks
            chunk = df.iloc[i:window+i,:]
            # calc_stats is a function created from the code above,

            beta,alpha,r2 = calc_stats_red(chunk)
            res.set_value(chunk.tail(1).index[0],"beta",beta)
            res.set_value(chunk.tail(1).index[0],"alpha",alpha)
            res.set_value(chunk.tail(1).index[0],"r2",r2)

    res = res.dropna()
    return res
red=pd.DataFrame(rolling_stats_r(df))

# simple regession without rolling for Green
import statsmodels.api as sm


def calc_stats_g(df):
    X = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(df.iloc[:, 3], X).fit()

    beta = model.params["Alpha"]
    alpha = model.params["const"]
    r2 = model.rsquared
    return beta, alpha, r2


calc_stats_g(df)
def rolling_stats_g(df, window=501):
    #     dataframe to hold the results
    res = pd.DataFrame(index=df.index)

    for i in range(0,len(df.index)):

        if len(df) - i >= window:
            # break the df into smaller chunks
            chunk = df.iloc[i:window+i,:]
            # calc_stats is a function created from the code above,

            beta,alpha,r2 = calc_stats_g(chunk)
            res.set_value(chunk.tail(1).index[0],"beta",beta)
            res.set_value(chunk.tail(1).index[0],"alpha",alpha)
            res.set_value(chunk.tail(1).index[0],"r2",r2)

    res = res.dropna()
    return res
green=pd.DataFrame(rolling_stats_g(df))

# simple regession without rolling for Blue
import statsmodels.api as sm


def calc_stats_b(df):
    X = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(df.iloc[:, 4], X).fit()

    beta = model.params["Alpha"]
    alpha = model.params["const"]
    r2 = model.rsquared
    return beta, alpha, r2


def rolling_stats_b(df, window=501):
    res = pd.DataFrame(index=df.index)


    for i in range(0, len(df.index)):

        if len(df) - i >= window:
            # break the df into smaller chunks
            chunk = df.iloc[i:window + i, :]
            # calc_stats is a function created from the code above,

            beta, alpha, r2 = calc_stats_b(chunk)
            res.set_value(chunk.tail(1).index[0], "beta", beta)
            res.set_value(chunk.tail(1).index[0], "alpha", alpha)
            res.set_value(chunk.tail(1).index[0], "r2", r2)


    res = res.dropna()
    return res
blue=rolling_stats_b(df)

# Task 3
r2 = pd.DataFrame(index=[i for i in range(6700)], columns= ["Red", "Green", "Blue"])
x = r2.index
y1 = red['r2']
y2 = green['r2']
y3=blue["r2"]


plt.plot(x,y1,'-r')
plt.plot(x,y2,'-g')
plt.plot(x,y3,'-b')
plt.axhline(y=0.5, color='r', linestyle='-',linewidth=0.4)
plt.show()

# I defined that bad correlation occurs when R-square is below 0.5. From the plot, it seems that for red，bad correlation
#occur between 1017 seconds and 1611 seconds. For Green one, bad correlation between 705 and 1670. For Blue, the bad
#correlation show up between 1106 and 1700, also between 2146 and 2890 seconds.

# Task 4
import threading
import time
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def getRsquare(predictor, color):
    predictor = pd.DataFrame(predictor)
    color = pd.DataFrame(color)
    result = []
    for i in range(500, color.shape[0]):
        rolling_predictor = predictor.iloc[(i - 500): i, :]
        rolling_color = color.iloc[(i - 500): i, :]
        regression = LinearRegression()
        regression.fit(rolling_color, rolling_predictor)
        predict_color = regression.predict(rolling_color)
        r2 = r2_score(rolling_predictor, predict_color)

        result.append(r2)
    return result


r_square = pd.DataFrame(index=[i for i in range(6700)], columns=["Red", "Green", "Blue"])


def ParallelRegression(threadnum):
    for i in range(threadnum):
        r_square.iloc[:, i] = getRsquare(df.Alpha, df[df.columns[i + 2]])


start = time.time()
threads = []
for threadnum in range(4):
    t = threading.Thread(target=ParallelRegression, args=(threadnum,))
    threads.append(t)
    t.start()

for threadnum in range(4): threads[threadnum].join()



#plt.figure()
plt.plot(r_square.index, r_square.Red, 'r-')
plt.plot(r_square.index, r_square.Green, 'g-')
plt.plot(r_square.index, r_square.Blue, 'b-')
plt.axhline(y=0.5, color='r', linestyle='-',linewidth=0.4)
plt.show()
print(r_square.head())
# Two plots are the same! Two methods get the same result

# Task 5

def confidence(predictor,color ):
    predictor = np.array(predictor).reshape(-1, 1)
    color= np.array(color).reshape(-1, 1)
    result = []
    for i in range(500, predictor.shape[0]):
        new_predictor = predictor[(i - 500) : i]
        new_color= color[(i - 500) : i]
        lr = LinearRegression()
        lr.fit(new_color, new_predictor)
        predict_val = lr.predict(new_color)
        sse = (new_predictor - predict_val) ** 2
        mean_sse = np.mean(sse)
        sd_sse = np.var(sse)**0.5
        interval = 2 * (mean_sse + 1.96 * sd_sse)
        result.append(interval)
    return result


confi_interval = pd.DataFrame(index=[i for i in range(6700)], columns= ["Red", "Green", "Blue"])


def ParallelRegression_con(threadnum):
    for i in range(threadnum):
        confi_interval.iloc[:, i] = confidence(df.Alpha, df[df.columns[i + 2]])

threads = []
for threadnum in range(4):
    t = threading.Thread(target=ParallelRegression_con, args=(threadnum,))
    threads.append(t)
    t.start()

for threadnum in range(4): threads[threadnum].join()


plt.plot(confi_interval.index, confi_interval.Red, 'r-')
plt.plot(confi_interval.index, confi_interval.Green, 'g-')
plt.plot(confi_interval.index, confi_interval.Blue, 'b-')
plt.axhline(y=2300, color='y', linestyle='-',linewidth=0.4)
plt.show()

# From this plot we can see that, bad correlation occur around the same time.