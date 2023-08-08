# %% [markdown]
# # **Mini Project Investigate Hotel Business using Data Visualization**

# %% [markdown]
# 

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("D:/DATA SCIENCE BOOTCAMP/Mini Project Investigate Hotel Business/hotel_bookings_data.csv")
df.head()

# %% [markdown]
# ## Data Preprocessing

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
df.nunique()

# %% [markdown]
# ## Handling Missing Value

# %%
df_clean = df.copy()

# %%
missing_value = df_clean.isna().sum() *100/len(df)
print(round(missing_value, 4).sort_values(ascending=False))

# %%
df_clean['company'] = df_clean['company'].fillna(0)
df_clean['agent'] = df_clean['agent'].fillna(0)
df_clean['children'] = df_clean['children'].fillna(0)
df_clean['total_guests'] = df_clean['total_guests'].fillna(0)
df_clean['city'] = df_clean['city'].fillna('unknown')


# %% [markdown]
# ## Changes Unsuitable Value

# %%
df['meal'].value_counts()

# %%
df_clean = df_clean.replace({'meal': {'Undefined':'No Meal'}})
df_clean['meal'].value_counts()

# %%
df_clean['children'] = df_clean['children'].astype('int64')
df_clean['agent'] = df_clean['agent'].astype('int64')
df_clean['company'] = df_clean['company'].astype('int64')
df_clean['total_guests'] = df_clean['total_guests'].astype('int64')

# %%
df_clean.info()

# %%
df_clean['total_guests'] = df_clean['adults'] + df_clean['children'] + df_clean['babies']
print('Data with 0 guest: {0} out of {1} all data'.format(df_clean[df_clean['total_guests'] == 0].shape[0], df_clean.shape[0]))
df_clean['stay_duration'] = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_weekdays_nights']
print('Data with 0 night: {0} out of {1} all data'.format(df_clean[df_clean['stay_duration'] == 0].shape[0], df_clean.shape[0]))

# Pick the necessary data
df_clean_fix = df_clean[(df_clean['total_guests'] > 0) & (df_clean['stay_duration'] > 0)]

print('before pre-processing:', df_clean.shape[0])
print('after pre-processing:', df_clean_fix.shape[0])

# %%
df_final = df_clean_fix.copy()

# %%
df_final.head()

# %% [markdown]
# ## Monthly Booking Analysis

# %%
df_final.groupby(['hotel', 'arrival_date_month', 'arrival_date_year', 'is_canceled']).agg({'is_canceled':'count'})

# %%
# creat total of booking (month, year)
df_monthly = df_final.groupby(['hotel', 'arrival_date_month', 'arrival_date_year']).agg({'is_canceled':'count'}).reset_index()

# ammount of year/hotel and month
df_monthly_year = df_monthly.groupby(['hotel', 'arrival_date_month']).agg({'arrival_date_year':'count'})
df_monthly= df_monthly.merge(df_monthly_year, on = ['hotel', 'arrival_date_month'])
df_monthly.rename(columns={'is_canceled':'total_booking', 'arrival_date_year_x':'arrival_date_year', 'arrival_date_year_y':'ammount_of_year'}, inplace=True)

# total of booking 
df_monthly_sum = df_monthly.groupby(['hotel', 'arrival_date_month']).agg({'total_booking':'sum'})
df_monthly_sum.rename(columns={'total_booking':'sum_booking'}, inplace=True)
df_monthly = df_monthly.merge(df_monthly_sum, on=['hotel', 'arrival_date_month'])

#create average num booking 
df_monthly['avg_num_booking'] = round(df_monthly['sum_booking']/df_monthly['ammount_of_year'])

df_monthly.sample(5)

# %%
table1 = pd.pivot_table(data=df_monthly, 
                        index=['arrival_date_month'], 
                        columns=['hotel'], 
                        values='avg_num_booking')
table1

# %%
df_monthly_final = df_monthly.copy()
df_monthly_final = df_monthly_final.replace({'arrival_date_month': {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 
                                            'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}})
table1 = pd.pivot_table(data=df_monthly_final, 
                        index=['arrival_date_month'], 
                        columns=['hotel'], 
                        values='avg_num_booking')
table1

# %%
#selecting important column to create percentage column
df_monthly_pct = df_monthly_final[['hotel', 'arrival_date_month', 'avg_num_booking']]

#there're duplciated data
df_monthly_pct = df_monthly_pct.drop_duplicates()

#summarize total of avg num booking based on hotel type
df_part = df_monthly_pct.groupby('hotel').agg({'avg_num_booking':'sum'})
df_part.rename(columns={'avg_num_booking':'total_avg_num_booking'}, inplace=True)
df_monthly_pct = df_monthly_pct.merge(df_part, on='hotel')

#create percentage column based on avg 
df_monthly_pct['percentage'] = round((df_monthly_pct['avg_num_booking']/df_monthly_pct['total_avg_num_booking'])*100, 2)
df_monthly_pct.sort_values('arrival_date_month', ascending=True)

# %%
fig, ax = plt.subplots(figsize=(17, 6))
plt.title("Average Number of Hotel Bookings per Month\nBased on Hotel Types", fontsize=15, color='black', weight='bold', pad=65)
plt.text(x=-1.5, y=5250, s="Juni dan Juli adalah musim puncak untuk pemesanan hotel.\nHotel kota mencapai jumlah rata-rata pemesanan hotel tertinggi sebesar 11,18% pada bulan Juli dan hotel resor mencapai 9,82% pada bulan Juni.\nAda lagi pertumbuhan rata-rata jumlah pemesanan hotel pada bulan Desember untuk hotel kota (10,32%) dan hotel resor (9,61%).\nHal ini dapat disebabkan oleh liburan Natal dan Malam Tahun Baru", fontsize=12, fontstyle='italic')

sns.barplot(x='arrival_date_month', y='avg_num_booking', data=df_monthly_final, hue='hotel', edgecolor='black', palette='pastel')

plt.xlabel('Month(s)', fontsize=11)
plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.ylabel('Average Number of Booking', fontsize=11)
plt.ylim(0, 5000)

plt.bar_label(ax.containers[0], padding=2)
plt.bar_label(ax.containers[1], padding=2)

plt.axvline(4.5, ls='--', color='green')
plt.axvline(6.5, ls='--', color='green')
plt.stackplot(np.arange(4.5, 7.5), [[5000]], color='limegreen', alpha=0.3)
plt.text(x=4.95, y=4750, s='Peak Season', fontsize=14, color='green', va='center')

plt.axvline(10.5, ls='--', color='red')
plt.axvline(11.5, ls='--', color='red')
plt.stackplot(np.arange(10.5, 12.5), [[5000]], color='indianred', alpha=0.3)
plt.text(x=10.7, y=4650, s='Holiday\nSeason', fontsize=14, color='red', va='center')

plt.legend(title='Hotel', fontsize=11, title_fontsize=12)

plt.bar_label(ax.containers[0], ['5.88%', '5.59%', '5.07%', '6.71%', '8.76%', '10.18%', '11.18%', '10.75%', '7.33%', '8.15%', '10.07%', '10.32%'], label_type='center', color='white', weight='bold', fontsize=8)
plt.bar_label(ax.containers[1], ['6.58%', '7.09%', '5.91%', '8.43%', '9.04%', '9.82%', '9.63%', '8.25%', '8.27%', '8.87%', '8.48%', '9.61%'], label_type='center', color='white', weight='bold', fontsize=8)

sns.despine()
plt.tight_layout()
plt.savefig('avg_num_booking.png', dpi=200)
plt.show()

# %% [markdown]
# ## Impact Analysis of Stay Duration on Hotel Bookings Cancellation Rates

# %%
df_final['stay_duration'].value_counts()

# %%
duration_list = []
for i in df_final['stay_duration']:
    if i >= 1 and i <= 7:
        group = '1 Week'
    elif i >= 8 and i <= 14:
        group = '2 Weeks'
    elif i >= 15 and i <= 21:
        group = '3 Weeks'
    else: 
        group = '4 Weeks'
    duration_list.append(group)
df_final['stay_duration_group'] = duration_list

# %%
df_final['stay_duration_group'].value_counts()

# %%
df_task3 = df_final.groupby(['hotel', 'stay_duration_group', 'is_canceled']).agg({'agent':'count'}).reset_index()
df_task3.rename(columns={'agent':'num_booking'}, inplace=True)

#create sum booking column
df_sum = df_task3.groupby(['hotel', 'stay_duration_group']).agg({'num_booking':'sum'}).reset_index()
df_sum.rename(columns={'num_booking':'sum_booking'}, inplace=True)
df_task3 = df_task3.merge(df_sum, on=['hotel', 'stay_duration_group'])

#create cancellation rate
df_task3['cancellation_rate'] = round((df_task3['num_booking']/df_task3['sum_booking'])*100, 2)
df_task3

# %%
#pick the data that hotels has been cancelled
df_task3_plot = df_task3[df_task3['is_canceled']==1].sort_values('stay_duration_group', ascending=True)
df_task3_plot

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Data dan plot sebelumnya
fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Tren Positif Tingkat Pembatalan Pemesanan Hotel\nper Durasi Menginap Berdasarkan Jenis Hotel", fontsize=15, color='black', weight='bold', pad=65)
plt.text(x=-1, y=105, s="Semakin lama pelanggan menginap, semakin tinggi persentase pemesanan yang dibatalkan\nPemesanan hotel hotel kota yang paling banyak dibatalkan adalah pada durasi menginap empat minggu (87,23%).\nPemesanan hotel resor yang paling banyak dibatalkan adalah pada durasi menginap tiga minggu (46,75%)", fontsize=12, fontstyle='italic')

sns.barplot(x='stay_duration_group', y='cancellation_rate', data=df_task3_plot, hue='hotel', edgecolor='black', palette='pastel')

sns.regplot(x=np.arange(0, len(df_task3_plot[df_task3_plot['hotel'] == 'City Hotel'])), y='cancellation_rate', 
            data=df_task3_plot[df_task3_plot['hotel'] == 'City Hotel'], scatter=False, label='Trend City Hotel', truncate=False, color='blue')
sns.regplot(x=np.arange(0, len(df_task3_plot[df_task3_plot['hotel'] == 'City Hotel'])), y='cancellation_rate', 
            data=df_task3_plot[df_task3_plot['hotel'] == 'Resort Hotel'], scatter=False, label='Trend Resort Hotel', truncate=False, color='orange')

plt.xlabel('Stay Duration(s)', fontsize=11)
plt.ylabel('Cancellation Rate(%)', fontsize=11)
plt.ylim(0, 100)

plt.bar_label(ax.containers[0], padding=5)
plt.bar_label(ax.containers[1], padding=2)

plt.legend(title='Hotel', fontsize=11, title_fontsize=12)

sns.despine()
plt.tight_layout()
plt.savefig('cancelrate_stayduration.png', dpi=200)
plt.show()


# %% [markdown]
# ## Impact Analysis of Lead Time on Hotel Bookings Cancellation Rate

# %%
df_final['lead_time'].value_counts()

# %%
lead_time_list=[]
for i in df_final['lead_time']:
    if i <= 30:
        lead_group = '1 Month'
    elif i >= 31 and i <= 120:
        lead_group = '2-4 Months'
    elif i >= 121 and i <= 210:
        lead_group = '5-7 Months'
    elif i >= 211 and i <= 300:
        lead_group = '8-10 Months'
    elif i >= 311 and i <= 360:
        lead_group = '11-12 Months'
    else: 
        lead_group = '>12 Months'
    lead_time_list.append(lead_group)
df_final['lead_time_group'] = lead_time_list

# %%
df_final['lead_time_group'].value_counts()

# %%
df_task4 = df_final.groupby(['hotel', 'lead_time_group', 'is_canceled']).agg({'agent':'count'}).reset_index()
df_task4.rename(columns={'agent':'num_booking'}, inplace=True)

#create sum booking column
df_sum = df_task4.groupby(['hotel', 'lead_time_group']).agg({'num_booking':'sum'}).reset_index()
df_sum.rename(columns={'num_booking':'sum_booking'}, inplace=True)
df_task4 = df_task4.merge(df_sum, on=['hotel', 'lead_time_group'])

#create cancellation rate
df_task4['cancellation_rate'] = round((df_task4['num_booking']/df_task4['sum_booking'])*100, 2)
df_task4

# %%
#pick the data that hotels has been cancelled
df_task4_plot = df_task4[df_task4['is_canceled']==1]
df_task4_plot

# %%
# Custom color palette for the plot
custom_palette = 'Set2'

fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Cancellation Rate of Hotel Bookings per Lead Time\nBased on Hotel Types", fontsize=15, color='black', weight='bold', pad=50)
plt.text(x=-1, y=105, s="Both hotel types have the lowest cancellation rate on 1-month lead time (city = 22.47%; resort = 13.11%)\nand the highest cancellation rate on 11-12 months lead time (city = 77.41%; resort = 43.5%).", fontsize=12, fontstyle='italic')

# Change the palette to custom_palette
sns.barplot(x='lead_time_group', y='cancellation_rate', data=df_task4_plot, hue='hotel', order=['1 Month', '2-4 Months', '5-7 Months', '8-10 Months', '11-12 Months', '>12 Months'], edgecolor='black', palette=custom_palette)

plt.xlabel('Lead Time(s)', fontsize=11)
plt.ylabel('Cancellation Rate(%)', fontsize=11)
plt.ylim(0, 100)

plt.bar_label(ax.containers[0], padding=2)
plt.bar_label(ax.containers[1], padding=2)

plt.axvline(3.5, ls='--', color='red')
plt.axvline(4.5, ls='--', color='red')
plt.stackplot(np.arange(3.5, 5.5), [[100]], color='indianred', alpha=0.3)
plt.text(x=3.75, y=95, s='Highest', fontsize=14, color='red', va='center')

plt.axvline(0.5, ls='--', color='green')
plt.axvline(-0.5, ls='--', color='green')
plt.stackplot(np.arange(-0.5, 1.5), [[100]], color='green', alpha=0.3)
plt.text(x=-0.25, y=95, s='Lowest', fontsize=14, color='green', va='center')

plt.legend(title='Hotel Type', prop={'size': 8}, loc=1)

sns.despine()
plt.tight_layout()
plt.savefig('cancelrate_leadtime.png', dpi=200)

plt.show()

# %%



