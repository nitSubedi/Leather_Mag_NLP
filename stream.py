import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df=pd.read_csv('Leather_Mag_Text_Cleaned.csv')


header = st.container()
dataset = st.container()


with header:
    st.title("Analyzing news articles")

with dataset:
    st.text("Provided data preview")
    st.write(df.head(10))   
    
    st.text("Cleaning up the data and showing the new data")
    df=pd.read_csv('data.csv')


    st.write(df.head(5))

df['Dates'] = pd.to_datetime(df['Dates'], format='%d %B %Y')


articles_per_day = df.groupby(df['Dates'].dt.date).size()




def generate_year_range_and_graph(df):
   
    st.title("Year Range Bar Graphs you to generate a bar graph based on a range of years.")

    start_year = st.number_input("Select start year:", value=2020, min_value=1900, max_value=2100)
    end_year = st.number_input("Select end year:", value=2021, min_value=1900, max_value=2100)

    filtered_df = df[(df['Dates'].dt.year >= start_year) & (df['Dates'].dt.year <= end_year)]

    article_counts = filtered_df['Dates'].dt.year.value_counts().sort_index()

    if start_year <= end_year:
        st.bar_chart(article_counts)
    else:
        st.error("Error: End year must be after or equal to start year.")




df['Dates'] = pd.to_datetime(df['Dates'])

generate_year_range_and_graph(df)

st.text("Find the articles in the year")
year = st.number_input("Enter the year:", value=2020, min_value=1900, max_value=2100)

if st.button("Find Articles"):
    articles_in_year = df[df['Dates'].dt.year == year]

    st.write(f"Articles in {year}:")
    for index, row in articles_in_year.iterrows():
        st.write("Title:", row['Titles'])
        st.write("News:", row['News'])
        st.write("-----------")

st.text("Find the Potential Fake articles in the year")
year2 = st.number_input("Enter the year:", value=2020, min_value=1901, max_value=2101)
if st.button("Find Potential Fake Articles"):
     fake_articles_in_year = df[(df['News_Type'] == 'Fake') & (df['Dates'].dt.year == year)]
     st.write(f"Articles in {year}:")
     for index, row in fake_articles_in_year.iterrows():
        st.write("Title:", row['Titles'])
        st.write("News:", row['News'])
        st.write("-----------")   




st.title("Understanding polarity and subjectivity")
kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['Polarity', 'Subjectivity']])


df['Cluster'] = kmeans.labels_

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Polarity'],df['Subjectivity'], c=df['Cluster'], cmap='viridis', alpha=0.5)
ax.set_title('News Articles Clustering based on Polarity and Subjectivity')
ax.set_xlabel('Polarity')
ax.set_ylabel('Subjectivity')
plt.colorbar(scatter, ax=ax, label='Cluster')
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='X')  # Marking the centroids


st.pyplot(fig)

st.text("1.The closer subjectivity value is to 1. The less factual it is estimated to be.\nMaking it far more likely to be fake")
st.text("2.For polarity, the closer the values are to -1 (extremely negative review) or 1 (extremely positive review) the more likely it is to be false")

df['Year'] = df['Dates'].dt.year

# Title and description
st.title("Total Number of Fake Articles by Year")
st.write("This app displays a line graph showing the total number of fake articles for each year.")

# Year range selection
start_year = st.slider("Select Start Year", min_value=min(df['Year']), max_value=max(df['Year']), value=min(df['Year']))
end_year = st.slider("Select End Year", min_value=min(df['Year']), max_value=max(df['Year']), value=max(df['Year']))

# Filter DataFrame to include only the selected range of years
filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Filter DataFrame to include only fake articles
fake_articles = filtered_df[filtered_df['News_Type'] == 'Fake']

# Group by year and count the number of fake articles
fake_articles_by_year = fake_articles.groupby('Year').size()

# Plot line graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fake_articles_by_year.index, fake_articles_by_year.values, marker='o', linestyle='-')
ax.set_xlabel("Year")
ax.set_ylabel("Total Number of Fake Articles")
ax.set_title("Total Number of Fake Articles by Year")

# Display plot using Streamlit
st.pyplot(fig)