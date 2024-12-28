import pandas as pd
from langdetect import detect, LangDetectException
import time

# This option let to view all the dataframe's columns
pd.set_option('display.max_columns', None)

# Uploading web scraped data
data_raw = pd.read_csv('C:\\Users\\el ruchenzo\\jobsproject\\jobsproject\\cvbankas_20241008_encoding_sutvarkytas - suklasifikuotas ranka.csv')
print(data_raw.columns)


# RAW DATASET INSPECTION 

# Checking for DUPLICATES
print(data_raw.duplicated().sum())  # Count of identical rows which is 0, because raw dataset contains ids of job ads

# Looping through each column and check for duplicates
for col in data_raw.columns:
    duplicates = data_raw[col].duplicated().sum()  # Count of duplicates
    print(f"Column '{col}' has {duplicates} duplicate(s).")

# Leaving only text columns and the manually given code from classificator
df = data_raw[['title', 'description', 'code']].copy()

# Checking for identical rows
print(df.duplicated().sum())  # Count of identical rows which isn't 0 anymore, because job ads ids were removed and
                              # some companies look for same profession employees in a few cities at the same time

# Checking if there are any NA values
print("NA values per column:", df.isna().sum()) # None


# LANGUAGE DETECTION

# df.loc[:, 'language'] = df['title'].apply(detect)
# unique languages found (31): lt, de, tr, id, it, pt, en, tl, hu, sw, af, vi, sv, ca, et, lv, es, nl, hr, sl, ro, ru, fr,
#                              so, no, fi, pl, sq, cy, da, sk.
start_time = time.time()
df['language'] = df['description'].apply(detect)
end_time = time.time()
print(f"Language detection time: {end_time - start_time:.2f} seconds")
# unique languages found (5): lt, pl, en, ru, de.

# Further, for language detection step I used 'description' column instead of 'title', because it has more text/context and
# the function determines language way more accuratelly than when using 'title' column. Additionally, I manually checked the
# results, and can confirm that the languages of job ads were determined correctly.

# Count occurrences of each language
language_counts = df['language'].value_counts()
print(language_counts)

# job ads with detected ru / pl languages contained lt text as well, but since there were only 23 of them for further work
# I removed them just to avoid any misunderstandings.

# Filtering only lithuanian job ads
df_lt = df[df['language'] == 'lt'].copy() # 8176 rows in total
#df_lt.drop(columns=['language'], inplace=True)
df_lt = df_lt[['title', 'description', 'code']].copy()
df_lt = pd.DataFrame(df_lt)
print(df_lt.columns)
print(df_lt.dtypes)
df_lt['title'] = df_lt['title'].astype(str)
df_lt['description'] = df_lt['description'].astype(str)
print(df_lt.dtypes)
print('done')

df_lt.to_csv('C:\\Users\\el ruchenzo\\jobsproject\\jobsproject\\lt_data.csv', index=False, encoding='utf-8')
