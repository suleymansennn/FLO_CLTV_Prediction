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
    - CRM müşteriler ile kurulucak olan iletişimi farklı açılardan, yüksek farkındalıklar ile gerçekleştirme imkanı 
    sağlar.
    - Crm analitiği çalışmaları ile amacımız tüm müşteri ilişkileri sürecini veriye dayalı olarak daha verimli hale 
    getimeye çalışmaktır.
    - Bu proje ile CLTV(Customer LifeTime Value) tahmini yaparak bir CRM uygulaması yapmış olacağız.
"""

"""
CLTV 
    - CLTV bir müşterinin şirkette kaldığı sürece şirkete bırakacağı toplam para değeridir. Yani bir müşterinin şirketle
    olan iletişimi süresince şirkete vereceği parasal değerdir.
    - Yani aslında müşterilerin geçmişteki davranışlarına bakarak gelecek hakkında bir önsezi elde edilir.
"""

"""
Customer LifeTime Value Prediction 
    - Zaman projeksiyonlu olasılıksal lifetime value tahmini yapılır.
    - Bütün kitlenin satın alma davranışları olasılıksal olarak modellenir. Daha sonra modeli koşullu, kişi özelinde 
    biçimlendirerek beklenen satın almayı tahminleyeceğiz.
    
    CLTV = Purchase Frequency * Average Order Value
    CLTV = Expected Number of Transaction * Expected Average Profit
    CLTV = BG/NBD Model * Gamma Gamma Submodel
"""

"""
İş Problemi:
    FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin orta uzun vadeli plan
    yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi
    gerekmektedir.
"""

"""
Veri Seti Hikayesi:
Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) olarak 
yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

    master_id: Eşsiz müşteri numarası
    order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
    last_order_channel: En son alışverişin yapıldığı kanal
    first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
    last_order_date: Müşterinin yaptığı son alışveriş tarihi
    last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
    last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
    order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
    order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
    customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
    customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
    interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
"""
# GÖREV 1: Veriyi Hazırlama

# Adım 1: flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv("crm/crm/flo_data_20k.csv")
df = df_.copy()
df.head()


# Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını
# tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile
# yuvarlayınız.
def outlier_threshold(dataframe, variable, q1=.05, q3=.95):
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
    print(f"{variable} variable has {len_outlier} outliers")
    # dataframe.loc[dataframe[variable]>up_limit] = round(up_limit)
    # dataframe.loc[dataframe[variable]<low_limit] = round(low_limit)
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

"""
    Veri Seti İle İlgili Aldığım Notlar:
        1. Tarih değişkenleri object olarak tanımlanmış. Tarih tipine değiştirilmesi gerekir.
        2. Veri setinde boş gözlem yok.
        4. Online alışveriş sayısı daha fazla
        5. Ödenen toplam ücretlerde de bir aykırılık var. Ayrıca ortalama değer ve medyan arasındaki fark sağa doğru 
        çarpıklık olduğunu gösteriyor.
        6. En son alışveriş yapılan kanal ile alışveriş kanalı arasında eşsiz değer farkı var?
            order_channel değişkeni sanırım müşterilerin üye olurken kullandığı kanal?
        7. Tüm müşteriler en az 1'er kere hem online hemde offline alışveriş yapmış
        8. Veri seti gruplanmış.  Bu yüzden yapılacak hesaplamalarda tekrardan bir gruplama yapılmasına gerek yok.
"""

# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
df.describe().T

for col in df.columns[df.columns.str.contains("total")]:
    replace_with_thresholds(df, col)

df.describe().T

# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz
df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

print(f'{df["new_total_purchases"].sum()} invoices were carried out from {df["first_order_date"].min()} to {df["last_order_date"].max()}')
print(f'{df["master_id"].nunique()} customer were served from {df["first_order_date"].min()} to {df["last_order_date"].max()}')

# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz
date_vars = df.columns[df.columns.str.contains("date")]

for col in date_vars:
    df[col] = pd.to_datetime(df[col])

# df[date_vars] = df[date_vars].apply(lambda x: pd.to_datetime(x))
df.info()

# Görev 2: CLTV Veri Yapısının Oluşturulması
# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
# Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir
# cltv dataframe'i oluşturunuz.

# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten
# ifade edilecek

# Tüm veri seti için hesaplama yapmadan önce bir müşteri özelinde gerekli değişkenleri oluşturalım.

dfx = df[df["master_id"] == "cc294636-19f0-11eb-8d74-000d3a38a36f"]
xmax_date = dfx["last_order_date"]
xmin_date = dfx["first_order_date"]

# recency: ilk alışverişi ile son alışverişi arasında geçen süre
print("customer minimum date:", xmin_date[0])
print("customer maximum date:", xmax_date[0])
xrec = (xmax_date - xmin_date).dt.days
print("recency:", xrec[0], "days")

# Tenure: müşterinin ilk alışverişinden analiz tarihine kadar geçen süre
xtoday = df["last_order_date"].max()
print("population maximum date:", xtoday + dt.timedelta(days=2))
xtenure = (xtoday - xmin_date[0]).days
print("T:", xtenure, "days")

# frequency: müşterinin toplam fatura sayısı
xfreq = dfx["new_total_purchases"]
print("frequency:", xfreq[0], "invoice")

# monetary: müşterinin fatura başına ortalama harcaması
xmone = dfx["new_total_expenditure"] / dfx["new_total_purchases"]
print("monetary:", xmone[0])



today_date = df["last_order_date"].max() + dt.timedelta(days=2)

cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]

cltv_df["recency"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7

cltv_df["T"] = (today_date - df["first_order_date"]).dt.days / 7

cltv_df["frequency"] = df["new_total_purchases"]
cltv_df["monetary"] = df["new_total_expenditure"] / df["new_total_purchases"]

cltv_df.head()
cltv_df.describe().T

# Adım 1: BG/NBD modelini fit ediniz.
"""
BG-NBD modeli kısaca belirli bir zaman aralığında bir müşterinin beklenen toplam satın alma sayısnı tahmin 
eder.(Fatura sayısı)

BG-NBD modeli satın alma sürecini(transaction process) ve markayı terk etme sürecini modeller.

Transaction Process(buy):
    Alive olduğu sürece, belirli bir zaman periyodunda, bir müşteri tarafından gerçekleştirilecek işlem sayısı 
    transaction rate parametresi ile poisson dağılır.
    Bir müşteri canlı olduğu sürece kendi transaction rate'i etrafında rastgele satın alma yapmaya devam edecektir.
    Transaction rate'ler her bir müşteriye göre değişir ve tüm kitle için gamma dağılır. (r, a)

Dropout Process (till you die):
    Her bir müşterinin, her satın alımından sonra  p olasılığı ile dropout rate'i vardır.
    Bir müşteri alışveriş yaptıktan sonra belirli bir olasılıkla drop olur.
    Dropout rate'ler her bir müşteriye göre değişir ve tüm kitle için beta dağılır (a, b)
"""
bgf = BetaGeoFitter(penalizer_coef=0.02)

# BG-NBD modeli frequency, recency ve Tenure deişkenleri üzerine fit edilir.
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

plot_period_transactions(bgf)
# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.


# Belirli bir zamana kadar beklenen satın alma sayısı.
cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])


cltv_df.sort_values("exp_sales_3_month", ascending=False).head(10)

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency"],
                                           cltv_df["T"])

cltv_df.head()

cltv_df.sort_values("exp_sales_6_month", ascending=False).head(10)
# 3 ve 6 aylık değerler sıralandığında aynı kişilerin tabloya girdiğini görüyoruz.
# Ayrıca 6 aylık değerler 3 aylık değerlerin 2 katı.

# Müşterilerin şirkette kalma olasılığı
bgf.conditional_probability_alive(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.

"""
Gamma Gamma Modeli
    BG-NBD modeli ile müşterilerin fatura sayısını tahmin etmiştirk. Geriye bu faturalara ortalama ne kadar para 
    harcayacaklarını tahmin etmek kalıyor bunu da Gamma Gamma modeli ile yapacağız. 
    Bir müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin etmek için kullanılır.
    Bir müşterinin yaptığı toplam harcama, transaction value'ların ortalaması etrafında rastgele dağılır.
    Ortalama transaction value, zaman içinde kullanıcılar arasında değişebilir fakat tek bir kullanıcı için değişmez.
    Ortalama transaction value tüm müşteriler arasında gamma dağılır.
    
    Model frequency ve monetary değerleri üzerine kurulur.
"""

ggf = GammaGammaFitter(penalizer_coef=0.02)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

# Modeli fit ettikten sonra ilgili fonksiyonu kullanarak her bir müşterinin beklenen harcamasını bulabiliriz.

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])
cltv_df.head()

# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv
# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values("cltv", ascending=False).head(20)

# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])

cltv_df.groupby("segment").agg({"count", "mean", "sum"})

sns.scatterplot(data=cltv_df, x="expected_average_profit", y="cltv", hue="segment")
plt.show(block=True)

# Segmentleri daha iyi ayırabilmek için değerleri çeyreklik değerlerine göre belirleyeceğim.
cltv_df["cltv"].describe([.1, .2, .3, .4, .5, .65, .7, .8, .9])
cltv_df["segment"] = pd.cut(cltv_df["cltv"], bins=[min(cltv_df["cltv"]),98,190,280,367,max(cltv_df["cltv"])],
                            labels=["D", "C", "B", "A2", "A1"])

sns.scatterplot(data=cltv_df, x="expected_average_profit", y="cltv", hue="segment")
plt.show(block=True)

cltv_df.groupby("segment").agg({"count", "mean", "sum"})