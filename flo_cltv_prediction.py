import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import matplotlib

matplotlib.use("Qt5Agg")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

"""
    - CRM stands for customer relationship management. It is a term used to describe the strategies, technologies, and practices that businesses use to manage and analyze customer interactions and data throughout the customer lifecycle. The goal of CRM is to improve customer relationships and business outcomes by organizing and automating business processes, such as sales, marketing, and customer service.
    - CRM systems typically include a range of tools and features that help businesses manage and analyze customer data, such as contact lists, customer interactions, sales data, and marketing performance. These tools can be used to automate and streamline a variety of business processes, such as lead generation, sales forecasting, and customer service.
    - CRM systems can be implemented in a variety of ways, including as on-premises software, as a cloud-based service, or as a combination of both. CRM systems can be used by businesses of all sizes and across a wide range of industries, and are often customized to meet the specific needs and goals of each organization.
"""

"""
Customer LifeTime Value Prediction 
    - Customer lifetime value (CLV) is a measure of the total value that a customer is expected to generate for a business over the course of their relationship with the company. It is often used to help businesses make decisions about marketing, customer acquisition, and customer retention efforts.
    - To calculate CLV, businesses typically consider factors such as the customer's average purchase value, the number of purchases they make over a given time period, and the length of time they are expected to remain a customer. CLV can be calculated for individual customers or for a group of customers.
    - Businesses use CLV to make informed decisions about how to allocate their resources. For example, a business may decide to invest more in acquiring and retaining customers with a high CLV, as these customers are expected to generate more revenue over the long term. Alternatively, a business may decide to focus on acquiring more customers with a lower CLV if they can do so at a lower cost.
    - Overall, CLV is a useful tool for businesses looking to optimize their marketing and customer acquisition efforts and maximize the value of their customer base.

    CLTV = Purchase Frequency * Average Order Value
    CLTV = Expected Number of Transaction * Expected Average Profit
    CLTV = BG/NBD Model * Gamma Gamma Submodel
"""

"""
Business Problem:
FLO would like to set a roadmap for sales and marketing activities. In order for the company to make a medium-long-term plan,
it is necessary to estimate the potential value that existing customers will provide to the company in the future.
"""

"""
Features
- master_id : Unique Customer Number
- order_channel : Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile)
- last_order_channel : The channel where the most recent purchase was made
- first_order_date : Date of the customer's first purchase
- last_order_channel : Customer's previous shopping history
- last_order_date_offline : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online : Total number of purchases made by the customer on the online platform
- order_num_total_ever_offline : Total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
- customer_value_total_ever_online :  Total fees paid for the customer's online purchases
- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
"""

df_ = pd.read_csv("crm/flo_data_20k.csv")
df = df_.copy()
df.head()


def outlier_threshold(dataframe, variable, q1=.01, q3=.99):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    len_outlier = len(dataframe.loc[dataframe[variable] > up_limit]) + len(
        dataframe.loc[dataframe[variable] < low_limit])
    print(f"{variable} variable have {len_outlier} outliers")
    dataframe[variable] = dataframe[variable].apply(
        lambda x: round(up_limit) if x > up_limit else (round(low_limit) if x < low_limit else x))


def check_df(dataframe, head=5, tail=5):
    print("*" * 70)
    print(" Shape ".center(70, "*"))
    print("*" * 70)
    print(dataframe.shape)

    print("*" * 70)
    print(" Types ".center(70, "*"))
    print("*" * 70)
    print(dataframe.dtypes)

    print("*" * 70)
    print(" Head ".center(70, "*"))
    print("*" * 70)
    print(dataframe.head(head))

    print("*" * 70)
    print(" Tail ".center(70, "*"))
    print("*" * 70)
    print(dataframe.tail(tail))

    print("*" * 70)
    print(" NA ".center(70, "*"))
    print("*" * 70)
    print(dataframe.isnull().sum())

    print("*" * 70)
    print(" Quantiles ".center(70, "*"))
    print("*" * 70)
    print(dataframe.describe([.01, .05, .1, .5, .9, .95, .99]).T)

    print("*" * 70)
    print(" Duplicate Rows ".center(70, "*"))
    print("*" * 70)
    print(dataframe.duplicated().sum())

    print("*" * 70)
    print(" Uniques ".center(70, "*"))
    print("*" * 70)
    print(dataframe.nunique())


check_df(df)

df.describe().T
"""
                                       count     mean      std     min      25%      50%      75%        max
order_num_total_ever_online       19945.0000   3.1109   4.2256  1.0000   1.0000   2.0000   4.0000   200.0000
order_num_total_ever_offline      19945.0000   1.9139   2.0629  1.0000   1.0000   1.0000   2.0000   109.0000
customer_value_total_ever_offline 19945.0000 253.9226 301.5329 10.0000  99.9900 179.9800 319.9700 18119.1400
customer_value_total_ever_online  19945.0000 497.3217 832.6019 12.9900 149.9800 286.4600 578.4400 45220.1300

"""
for col in df.columns[df.columns.str.contains("total")]:
    replace_with_thresholds(df, col)
"""
order_num_total_ever_online variable have 14 outliers
order_num_total_ever_offline variable have 20 outliers
customer_value_total_ever_offline variable have 17 outliers
customer_value_total_ever_online variable have 13 outliers
"""
df.describe().T
"""                                       count     mean      std     min      25%      50%      75%       max
order_num_total_ever_online       19945.0000   3.0920   3.8095  1.0000   1.0000   2.0000   4.0000   48.0000
order_num_total_ever_offline      19945.0000   1.8862   1.4347  1.0000   1.0000   1.0000   2.0000   16.0000
customer_value_total_ever_offline 19945.0000 251.9213 251.0237 10.0000  99.9900 179.9800 319.9700 3020.0000
customer_value_total_ever_online  19945.0000 489.7057 632.6098 12.9900 149.9800 286.4600 578.4400 7800.0000
"""

df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

print(
    f'{df["new_total_purchases"].sum()} invoices were carried out from {df["first_order_date"].min()} to {df["last_order_date"].max()}')
print(
    f'{df["master_id"].nunique()} customer were served from {df["first_order_date"].min()} to {df["last_order_date"].max()}')
"""
99289.0 invoices were carried out from 2013-01-14 to 2021-05-30
19945 customer were served from 2013-01-14 to 2021-05-30
"""

date_vars = df.columns[df.columns.str.contains("date")]

for col in date_vars:
    df[col] = pd.to_datetime(df[col])

# df[date_vars] = df[date_vars].apply(lambda x: pd.to_datetime(x))
df.info()
"""      
 0   master_id                          19945 non-null  object        
 1   order_channel                      19945 non-null  object        
 2   last_order_channel                 19945 non-null  object        
 3   first_order_date                   19945 non-null  datetime64[ns]
 4   last_order_date                    19945 non-null  datetime64[ns]
 5   last_order_date_online             19945 non-null  datetime64[ns]
 6   last_order_date_offline            19945 non-null  datetime64[ns]
 7   order_num_total_ever_online        19945 non-null  float64       
 8   order_num_total_ever_offline       19945 non-null  float64       
 9   customer_value_total_ever_offline  19945 non-null  float64       
 10  customer_value_total_ever_online   19945 non-null  float64       
 11  interested_in_categories_12        19945 non-null  object        
 12  new_total_expenditure              19945 non-null  float64       
 13  new_total_purchases                19945 non-null  float64       
"""
#####################################################################################
# Creating the CLTV Data Structure
#####################################################################################

dfx = df[df["master_id"] == "cc294636-19f0-11eb-8d74-000d3a38a36f"]
xmax_date = dfx["last_order_date"]  # 2021-02-26
xmin_date = dfx["first_order_date"]  # 2020-10-30
# Recency
print("customer minimum date:", xmin_date[0])  # minimum date: 2020-10-30 00:00:00
print("customer maximum date:", xmax_date[0])  # maximum date: 2021-02-26 00:00:00
xrec = (xmax_date - xmin_date).dt.days
print("recency:", xrec[0], "days")  # 119 days

# Tenure
xtoday = df["last_order_date"].max()
print("population maximum date:", xtoday + dt.timedelta(days=2))  # population maximum date: 2021-06-01 00:00:00
xtenure = (xtoday - xmin_date[0]).days
print("T:", xtenure, "days")  # T: 212 days

# frequency
xfreq = dfx["new_total_purchases"]
print("frequency:", xfreq[0], "invoice")  # frequency: 5.0 invoice

xmone = dfx["new_total_expenditure"] / dfx["new_total_purchases"]
print("monetary:", xmone[0])  # monetary: 187.874

today_date = df["last_order_date"].max() + dt.timedelta(days=2)

cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]

cltv_df["recency"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7

cltv_df["T"] = (today_date - df["first_order_date"]).dt.days / 7

cltv_df["frequency"] = df["new_total_purchases"]
cltv_df["monetary"] = df["new_total_expenditure"] / df["new_total_purchases"]

cltv_df.head()
"""
                            customer_id  recency        T  frequency  monetary
0  cc294636-19f0-11eb-8d74-000d3a38a36f  17.0000  30.5714     5.0000  187.8740
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f 209.8571 224.8571    21.0000   95.8833
2  69b69676-1a40-11ea-941b-000d3a38a36f  52.2857  78.8571     5.0000  117.0640
3  1854e56c-491f-11eb-806e-000d3a38a36f   1.5714  20.8571     2.0000   60.9850
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f  83.1429  95.4286     2.0000  104.9900
"""
cltv_df.describe().T
"""
               count     mean     std     min      25%      50%      75%       max
recency   19945.0000  95.2635 74.5894  0.0000  50.4286  76.5714 109.4286  433.4286
T         19945.0000 114.4718 74.7710  0.7143  73.8571  93.0000 119.4286  437.1429
frequency 19945.0000   4.9781  4.1209  2.0000   3.0000   4.0000   6.0000   57.0000
monetary  19945.0000 152.0990 73.5241 22.4900 103.6533 136.9062 182.4500 1401.8000
"""
#  Establishment of BG/NBD, Gamma-Gamma Models and calculation of CLTV

bgf = BetaGeoFitter(penalizer_coef=0.02)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

plot_period_transactions(bgf)
plt.show()

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])

cltv_df.sort_values("exp_sales_3_month", ascending=False).head(10)
"""
                                customer_id  recency        T  frequency  monetary  exp_sales_3_month
7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f  62.7143  67.2857    52.0000  166.2246             6.0296
15611  4a7e875e-e6ce-11ea-8f44-000d3a38a36f  39.7143  40.0000    29.0000  165.2976             4.6387
8328   1902bf80-0035-11eb-8341-000d3a38a36f  28.8571  33.2857    25.0000   97.4396             4.4092
6666   53fe00d4-7b7a-11eb-960b-000d3a38a36f   9.7143  13.0000    17.0000  259.8653             4.2859
14373  f00ad516-c4f4-11ea-98f7-000d3a38a36f  38.0000  46.4286    27.0000  141.3548             4.0171
19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f  52.5714  58.7143    31.0000  228.5300             4.0034
6756   27310582-6362-11ea-a6dc-000d3a38a36f  62.7143  64.1429    29.0000  168.8810             3.5649
10489  7af5cd16-b100-11e9-9757-000d3a38a36f 103.1429 111.8571    43.0000  157.1126             3.5511
10536  e143b6fa-d6f8-11e9-93bc-000d3a38a36f 104.5714 113.4286    40.0000  176.2000             3.2804
4315   d5ef8058-a5c6-11e9-a2fc-000d3a38a36f 133.1429 147.1429    49.0000  161.8467             3.2665
"""
# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])

cltv_df.sort_values("exp_sales_6_month", ascending=False).head(10)
"""
                                customer_id  recency        T  frequency  monetary  exp_sales_3_month  exp_sales_6_month
7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f  62.7143  67.2857    52.0000  166.2246             6.0296            12.0592
15611  4a7e875e-e6ce-11ea-8f44-000d3a38a36f  39.7143  40.0000    29.0000  165.2976             4.6387             9.2775
8328   1902bf80-0035-11eb-8341-000d3a38a36f  28.8571  33.2857    25.0000   97.4396             4.4092             8.8185
6666   53fe00d4-7b7a-11eb-960b-000d3a38a36f   9.7143  13.0000    17.0000  259.8653             4.2859             8.5718
14373  f00ad516-c4f4-11ea-98f7-000d3a38a36f  38.0000  46.4286    27.0000  141.3548             4.0171             8.0342
19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f  52.5714  58.7143    31.0000  228.5300             4.0034             8.0068
6756   27310582-6362-11ea-a6dc-000d3a38a36f  62.7143  64.1429    29.0000  168.8810             3.5649             7.1299
10489  7af5cd16-b100-11e9-9757-000d3a38a36f 103.1429 111.8571    43.0000  157.1126             3.5511             7.1022
10536  e143b6fa-d6f8-11e9-93bc-000d3a38a36f 104.5714 113.4286    40.0000  176.2000             3.2804             6.5608
4315   d5ef8058-a5c6-11e9-a2fc-000d3a38a36f 133.1429 147.1429    49.0000  161.8467             3.2665             6.5330
"""

bgf.conditional_probability_alive(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

ggf = GammaGammaFitter(penalizer_coef=0.02)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

# Modeli fit ettikten sonra ilgili fonksiyonu kullanarak her bir müşterinin beklenen harcamasını bulabiliriz.

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])
cltv_df.head()
"""
                            customer_id  recency        T  frequency  monetary  exp_sales_3_month  exp_sales_6_month  expected_average_profit
0  cc294636-19f0-11eb-8d74-000d3a38a36f  17.0000  30.5714     5.0000  187.8740             1.1849             2.3698                 197.7031
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f 209.8571 224.8571    21.0000   95.8833             1.0407             2.0815                  97.0980
2  69b69676-1a40-11ea-941b-000d3a38a36f  52.2857  78.8571     5.0000  117.0640             0.7042             1.4083                 123.4017
3  1854e56c-491f-11eb-806e-000d3a38a36f   1.5714  20.8571     2.0000   60.9850             0.7835             1.5670                  70.6300
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f  83.1429  95.4286     2.0000  104.9900             0.3526             0.7051                 120.4925
"""

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv", ascending=False).head(10)
"""
recency       T  frequency  monetary  exp_sales_3_month  exp_sales_6_month  expected_average_profit      cltv
6.1429 13.1429    11.0000  758.0855             2.9234             5.8468           774.8810           4753.6078
2.8571  7.8571     4.0000 1401.8000             1.4956             2.9912           1489.9907          4676.2169
9.7143 13.0000    17.0000  259.8653             4.2859             8.5718           263.6678           2371.3544
62.7143 67.2857    52.0000  166.2246             6.0296            12.0592          167.0311           2113.4048
3.4286 34.4286     8.0000  601.2262             1.6063             3.2127           619.7646           2089.1209
74.2857 74.5714    16.0000  501.8737             1.8811             3.7621          509.5241           2011.2524
68.2857  9.4286     2.0000  862.6900             0.9641             1.9282          979.0496           1980.7747
52.5714 58.7143    31.0000  228.5300             4.0034             8.0068          230.3629           1935.2711
16.1429  7.1429     9.0000  317.4844             2.7868             5.5737          326.3021           1908.2218
74.1429  8.8571     6.0000  431.3283             1.9548             3.9096          449.3741           1843.3385
"""

# Creating Segments
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])
cltv_df.groupby("segment").agg({"count", "mean", "sum"})

sns.scatterplot(data=cltv_df, x="expected_average_profit", y="cltv", hue="segment")
plt.show(block=True)

cltv_df["cltv"].describe([.1, .2, .3, .4, .5, .65, .7, .8, .9])
"""
count   19945.0000
mean      206.9206
std       162.1323
min         9.9399
10%        72.9221
20%        98.5865
30%       121.6650
40%       143.9613
50%       169.1092
65%       213.5990
70%       231.9317
80%       281.5491
90%       370.4602
max      4753.6078
"""
cltv_df["segment"] = pd.cut(cltv_df["cltv"], bins=[min(cltv_df["cltv"]), 98, 190, 280, 367, max(cltv_df["cltv"])],
                            labels=["D", "C", "B", "A2", "A1"])

sns.scatterplot(data=cltv_df, x="expected_average_profit", y="cltv", hue="segment")
plt.show(block=True)

cltv_df.groupby("segment").agg({"count", "mean", "sum"})
