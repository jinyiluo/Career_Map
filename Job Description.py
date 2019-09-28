# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:33:33 2019

@author: Jinyi Luo
"""
# Import all the packages
import pandas as pd
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import os
import nltk
import squarify
import matplotlib
import string
from textblob import TextBlob 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image


plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

folder = 'D:\\GWU\\Spring 2019\\DATS 6202\\Dataset\\jd\\results'

files = glob.glob(folder+"\\*.csv")  ## files is a list containing all the file names in the folder

jobs = pd.concat([pd.read_csv(f).assign(Category=os.path.basename(f).split('_')[2]) for f in files], ignore_index = True) # pd.concat() - returns one pd.DataFrame()

# Data Cleaning
# Check if there is any NaN
pd.isnull(jobs).sum()
 
# The value counts of each column
# Job Category
jobs.Category.value_counts()

# Job Titles
title=jobs.title.value_counts()
title_count=pd.DataFrame({'Title':title.index, 'Count':title.values})

# Company job posting
company=jobs.company.value_counts()
company_count=pd.DataFrame({'Company':company.index, 'Count':company.values})

# Plot the bar chart of Company
company_count[:10].plot.barh(x='Company', y='Count', legend=True)
plt.gca().invert_yaxis()
plt.title('Top 10 Most Opening Companies', fontsize=14)
plt.xlabel('Count')

# Post time
jobs.post_time.value_counts()

# Total numbers
total_no_company=jobs['company'].nunique()
print('Total number of vacancies firms',total_no_company)
total_no_location=jobs['location'].nunique()
print('Total number of vacancies location',total_no_location)
total_no_title=jobs['title'].nunique()
print('Total number of vacancies job titles',total_no_title)

#finding highest number of vacancy in a company
most_vacancy= jobs.groupby(['company'])['title'].count()
most_vacancy=most_vacancy.reset_index(name='title')
most_vacancy=most_vacancy.sort_values(['title'],ascending=False)
pareto_df=most_vacancy
most_vacancy=most_vacancy.head(25)
print('Top 10 firms with most vacancies',most_vacancy)

# Plot graph for top most vacancy
fig, ax = plt.subplots(figsize = (10,6))
ax=seaborn.barplot(x="company", y="title", data=most_vacancy)    
ax.set_xticklabels(most_vacancy['company'],rotation=90)  
ax.set_xlabel('Company with most jobs',fontsize=16, color='black')
ax.set_ylabel('Number of jobs',fontsize=16,color='black') 

# Extract Salary
Salary=pd.DataFrame(jobs['extra_info'].dropna()).apply(lambda x: x.str.replace(',',''))

time_list=['year','month','week','hour','day']
Salary['Time']=Salary['extra_info'].apply(lambda x : re.findall(r"(?=("+'|'.join(time_list)+r"))",x))

Job_Type=['Full-time','Part-time','Contract','Commission','Temporary','Internship','Other']
Salary['Job_Type']=Salary['extra_info'].apply(lambda x : re.findall(r"(?=("+'|'.join(Job_Type)+r"))",x))
Wage=pd.DataFrame(Salary[Salary.astype(str)['Job_Type'] == '[]'])
Wage['Wage']=Wage['extra_info'].apply(lambda x : re.findall(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]\d+)?)',x))


# Extract Minimum 
Wage['Minimum_Wage']=Wage['Wage'].apply(lambda m : min(m,key=float))
Wage['Minimum_Wage']=Wage.Minimum_Wage.astype(float)
# Then extract maximum years experience
Wage['Maximum_Wage']=Wage['Wage'].apply(lambda n : max(n,key=float))
Wage['Maximum_Wage']=Wage.Maximum_Wage.astype(float)

# Merge Salary & Wage
cols_to_use = Wage.columns.difference(Salary.columns)
Salary_Wage = pd.merge(Salary, Wage[cols_to_use], left_index=True, right_index=True, how='outer')
Salary_Wage['Average']=(Salary_Wage['Minimum_Wage']+Salary_Wage['Maximum_Wage'])/2
Wage.Minimum_Wage.describe()
Wage.Minimum_Wage.value_counts()
Wage.Maximum_Wage.describe()
Wage.Maximum_Wage.value_counts()

# Concatenate jobs_salary_wage
jobs_salary_wage = pd.concat([jobs, Salary_Wage], axis=1, join_axes=[jobs.index])

# Job Types
Job_Type=['Full-time','Part-time','Contract','Commission','Temporary','Internship']
Job_Types = dict((x,0) for x in Job_Type)
for i in Job_Type:
    x = jobs['extra_info'].str.contains(i).sum()
    if i in Job_Types:
        Job_Types[i]=x

print(Job_Types)    

# Job Salary
Job_Salary=['year','month','week','hour','day']
Job_Salaries = dict((x,0) for x in Job_Salary)
for i in Job_Salary:
    x = jobs['extra_info'].str.contains(i).sum()
    if i in Job_Salaries:
        Job_Salaries[i]=x

print(Job_Salaries)    

# Location posting
jobs['location']=pd.DataFrame(jobs['location']).apply(lambda x: x.str.replace('San Francisco Bay Area','San Francisco')).apply(lambda x: x.str.replace('Santa Clara Valley','Santa Clara')).apply(lambda x: x.str.replace('Township of Woodbridge','Woodbridge'))
jobs.location.value_counts()
location=jobs.location.value_counts()
location_count=pd.DataFrame({'Location':location.index, 'Count':location.values})

# Split location to City
location_count['City']=location_count['Location'].apply(lambda x : x.split(',')[-2])
# Split location to State
location_count['State']=location_count['Location'].apply(lambda x : x.split(',')[-1]).apply(lambda y : y.split(' ')[1])

# Split location to City
jobs['City']=jobs['location'].apply(lambda x : x.split(',')[-2])
# Split location to State
jobs['State']=jobs['location'].apply(lambda x : x.split(',')[-1]).apply(lambda y : y.split(' ')[1])
Location=pd.DataFrame({'City':jobs.City,'State':jobs.State})

# State job listing counts
jobs.State.value_counts()
# Top 10 City job listing counts 
jobs.City.value_counts()[:10]

# Data Science Category
Data_Science=jobs[jobs['Category']=='Data Scientist'].drop(columns='url')

# Split location to City
Data_Science['City']=Data_Science['location'].apply(lambda x : x.split(',')[-2])

# Split location to State
Data_Science['State']=Data_Science['location'].apply(lambda x : x.split(',')[-1]).apply(lambda y : y.split(' ')[1])
Data_Science_Salary_Wage=pd.concat([Data_Science, Salary_Wage], axis=1, join_axes=[Data_Science.index]).drop(columns=['extra_info','location','Category','Wage'])

# Data Science Salary Wage company count
Data_Science_Salary_Wage_company_count=pd.DataFrame({'Company':Data_Science_Salary_Wage.company.value_counts().index, 'Count':Data_Science_Salary_Wage.company.value_counts().values})

# Plot the bar chart of Company
Data_Science_Salary_Wage_company_count[:10].plot.barh(x='Company', y='Count', legend=True)
plt.gca().invert_yaxis()
plt.title('Top 10 Data Science Companies', fontsize=14)
plt.xlabel('Count')

# Data Science Salary
Data_Science_Salary=Data_Science_Salary_Wage.dropna(subset=['Minimum_Wage'])
# Convert list to String
Data_Science_Salary['Time'] = Data_Science_Salary['Time'].apply(lambda x: ''.join(map(str, x)))
# Get Data Scientists Yearly Salary
Data_Science_Yearly_Salary=Data_Science_Salary.loc[Data_Science_Salary.Time.str.contains('year')]
Data_Science_Yearly_Salary.Minimum_Wage.describe()
Data_Science_Yearly_Salary.Maximum_Wage.describe()

# Plot Minimum Data Scientists Yearly Salary
Data_Science_Yearly_Salary.Minimum_Wage.plot(kind='box')
plt.title('Minimum Data Scientists Yearly Salary')
plt.ylabel('Years')

# Use seaborn to plot Minimum Data Scientists Yearly Salary
sns.countplot('Minimum_Wage',data=Data_Science_Yearly_Salary)
plt.suptitle('Minimum Data Scientists Yearly Salary')


# Plot Maximum Data Scientists Yearly Salary
Data_Science_Yearly_Salary.Maximum_Wage.plot(kind='box')
plt.title('Maximum Data Scientists Yearly Salary')
plt.ylabel('Years')

# Use seaborn to plot Maximum Data Scientists Yearly Salary
sns.countplot('Maximum_Wage',data=Data_Science_Yearly_Salary)
plt.suptitle('Maximum Data Scientists Yearly Salary')


# Extract the year of work experience in each position
stop_words = set(stopwords.words('english'))
jobs['Responsibilities']=jobs.content.apply(lambda x : word_tokenize(x))
jobs['Responsibilities']=jobs.Responsibilities.apply(lambda x : [w for w in x if w not in stop_words])
jobs['Responsibilities']=jobs.Responsibilities.apply(lambda x : ' '.join(x))


# Exploratory
# Job Types for DS
DSJob_Type=['Full-time','Part-time','Contract','Commission','Temporary','Internship']
DSJob_Types = dict((x,0) for x in DSJob_Type)
for i in DSJob_Type:
    x = Data_Science['extra_info'].str.contains(i).sum()
    if i in DSJob_Types:
        DSJob_Types[i]=x

print(DSJob_Types)  

# Extract degree requirement of each role
Degree=['BA','BS','Bachelor','MBA', 'Master','MASTER','PhD','PHD','BA','B.A','BA/BS','MA/MS', 'MA','M.A', 'MS','M.S','Ph.D']

Degrees = dict((x,0) for x in Degree)   
for i in Degree:
    x = jobs['content'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i]=x

print(Degrees)


# Extract degree requirement of DS
Degree=['BA','BS','Bachelor','MBA', 'Master','MASTER','PhD','PHD']
#Degree=[x.upper() for x in ['BA','BS','Bachelor','MBA', 'Master','PhD']]
Degrees = dict((x,0) for x in Degree)   
for i in Degree:
    x = Data_Science_Salary_Wage['content'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i]=x

print(Degrees)

# print Degree table
degree_requirement = sorted(Degrees.items(), key=lambda x : x[1], reverse=True)
degree=pd.DataFrame(degree_requirement,columns=['Degree','Count'])
degree

# Plot the bar chart of Degree Distribution
degree.plot.barh(x='Degree', y='Count', legend=True)
plt.title('Degree Distribution', fontsize=14)
plt.xlabel('Count')

# find out which language occurs in most in minimum Qualifications string
wordcount = dict((x,0) for x in jobs['content'])
for w in re.findall(r"[\w'+#-]+|[.!?;’]", jobs['content']):
    if w in wordcount:
        wordcount[w] += 1
# print
print(wordcount)

# Check programing Languages R and C++
Programing_Languages =['Python','PYTHON','Java','JAVA','C/C+','Javascript','JAVASCRIPT',' Go ',' GO ',' R ', 'Swift','SWIFT','Php',' PHP', 'C#','MATLAB','Matlab','SAS','Sas','HTML','SQL']

Languages= dict((x,0) for x in Programing_Languages)   
for i in Languages:
    x = jobs['content'].str.contains(i).sum()
    if i in Languages:
        Languages[i]=x

print(Languages)  


# language Table
languages_requirement = sorted(Languages.items(),key= lambda x : x[1], reverse=True)
language=pd.DataFrame(languages_requirement,columns=['Language','Count'])
language['Count']=language.Count.astype('int')
language

# Language Plot
language.plot.barh(x='Language', y='Count', legend=False)
plt.suptitle('Language Distribution', fontsize=14)
plt.xlabel('Count')

# DS Programing_Languages
Programing_Languages =['Python','PYTHON','Java','JAVA','C/C+','Javascript','JAVASCRIPT',' Go ',' GO ',' R ', 'Swift','SWIFT','Php',' PHP', 'C#','MATLAB','Matlab','SAS','Sas','HTML','SQL']

DSLanguages= dict((x,0) for x in Programing_Languages)   
for i in Languages:
    x = Data_Science_Salary_Wage['content'].str.contains(i).sum()
    if i in DSLanguages:
        DSLanguages[i]=x

print(DSLanguages)                       

# DSlanguage Table
DSlanguages_requirement = sorted(DSLanguages.items(),key= lambda x : x[1], reverse=True)
DSlanguage=pd.DataFrame(DSlanguages_requirement,columns=['Language','Count'])
DSlanguage['Count']=DSlanguage.Count.astype('int')
DSlanguage

# DSLanguage Plot
DSlanguage.plot.barh(x='Language', y='Count', legend=False)
plt.suptitle('Data Scientist Language Requirements', fontsize=14)
plt.xlabel('Count')


    
# Tokenize content to Responsibilities
stop_words = set(stopwords.words('english'))
Data_Science_Salary_Wage['Responsibilities']=Data_Science_Salary_Wage.content.apply(lambda x : word_tokenize(x))
Data_Science_Salary_Wage['Responsibilities']=Data_Science_Salary_Wage.Responsibilities.apply(lambda x : [w for w in x if w not in stop_words])
Data_Science_Salary_Wage['Responsibilities']=Data_Science_Salary_Wage.Responsibilities.apply(lambda x : ' '.join(x))

# DS jobs Word Cloud
Res_AN=' '.join(Data_Science_Salary_Wage['Responsibilities'].tolist())

G = np.array(Image.open('D:\\GWU\\Spring 2019\\DATS 6202\\Python\\Human Head.jpg'))

sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=G,background_color="white").generate(Res_AN)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilities of Data Scientist',size=24)
plt.show()


# Analyst jobs
jobs_Analyst=jobs.loc[jobs.title.str.contains('Analyst').fillna(False)]
jobs_Analyst.location.value_counts()

Res_AN=' '.join(jobs_Analyst['Responsibilities'].tolist())
G = np.array(Image.open('D:\\GWU\\Spring 2019\\DATS 6202\\Python\\Human Head.jpg'))
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=G,background_color="white").generate(Res_AN)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilities',size=24)
plt.show()

# Intern jobs
Data_Science_Salary_Wage_Intern=Data_Science_Salary_Wage.loc[Data_Science_Salary_Wage.title.str.contains('Senior').fillna(False)]
Data_Science_Salary_Wage_Intern.State.value_counts()
Data_Science_Salary_Wage_Intern.City.value_counts()

Res_AN=' '.join(Data_Science_Salary_Wage_Intern['Responsibilities'].tolist())

G = np.array(Image.open('D:\\GWU\\Spring 2019\\DATS 6202\\Python\\Human Head.jpg'))

sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=G,background_color="white").generate(Res_AN)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilities',size=24)
plt.show()

# WordCloud Functions
def MadeWordCloud(title,text):
    jobs_subset=jobs.loc[jobs.title.str.contains(title).fillna(False)]
    long_text=' '.join(jobs_subset[text].tolist())
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    wordcloud =WordCloud(mask=G, background_color="white").generate(long_text)
    plt.figure()
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.margin(x=0,y=0)
    plt.title(text,size=24)
    plt.show()
    
MadeWordCloud('Data Scientist','Responsibilities')
MadeWordCloud('Data Science Intern','Responsibilities')
MadeWordCloud('Junior Data Scientist','Responsibilities')
MadeWordCloud('Senior Data Scientist','Responsibilities')
MadeWordCloud('Analyst','Responsibilities')
MadeWordCloud('Data Engineer','Responsibilities')
MadeWordCloud('Statistician','Responsibilities')



for i in Data_Science_Salary_Wage['Responsibilities']:
    analysis = TextBlob(i)

# Daraskills needed for Analyst position
DataSkill = [' R','Python','SQL','SAS','C#','Java']

DataSkills = dict((x,0) for x in DataSkill)
for i in DataSkill:
    x = jobs_Analyst['content'].str.contains(i).sum()
    if i in DataSkill:
        DataSkills[i] = x
        
print(DataSkills)

# Degrees needed for Analyst position
Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = jobs_Analyst['content'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)

# Minimum work experience needed for Analyst position
sns.countplot('Minimum_years_experience',data=jobs_Analyst)
plt.suptitle('Minimum work experience')


# Data Visualization Tools needed for a position
DV_Tools = ['Tableau','Power BI','Qlik','Data Studio','Google Analytics','GA']

DV = dict((x,0) for x in DV_Tools)
for i in DV:
    x = jobs['content'].str.contains(i).sum()
    if i in DV_Tools:
        DV[i] = x
        
print(DV)

# Statistical Analysis Tools needed for a position
SA_Tools = ['SPSS','R ','Matlab','Excel','Spreadsheet','SAS']

SA = dict((x,0) for x in SA_Tools)
for i in SA:
    x = jobs['content'].str.contains(i).sum()
    if i in SA_Tools:
        SA[i] = x
        
print(SA)


# Data Visualization Tools needed for DS position
DV_Tools = ['Tableau','Power BI','Qlik','Data Studio','Google Analytics','GA']

DV = dict((x,0) for x in DV_Tools)
for i in DV:
    x = Data_Science_Salary_Wage['content'].str.contains(i).sum()
    if i in DV_Tools:
        DV[i] = x
        
print(DV)

# Statistical Analysis Tools needed for DS position
SA_Tools = ['SPSS','R ','Matlab','Excel','Spreadsheet','SAS']

SA = dict((x,0) for x in SA_Tools)
for i in SA:
    x = Data_Science_Salary_Wage['content'].str.contains(i).sum()
    if i in SA_Tools:
        SA[i] = x
        
print(SA)

# Positions in the New York
jobs_NY = jobs.loc[jobs.City == 'New York']

jobs_NY_Type = jobs_NY.Category.value_counts()
jobs_NY_Type = jobs_NY_Type.rename_axis('Type').reset_index(name='counts')
jobs_NY_Type

# Squarify plot
matplotlib.rcParams.update({'font.size': 8})
cmap = matplotlib.cm.Blues
norm = matplotlib.colors.Normalize(vmin=min(jobs_NY_Type.counts), vmax=max(jobs_NY_Type.counts))
colors = [cmap(norm(value)) for value in jobs_NY_Type.counts]
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(24, 6)
squarify.plot(sizes=jobs_NY_Type['counts'], label=jobs_NY_Type['Type'], alpha=.8, color=colors)
plt.title('Type of positions',fontsize=20,fontweight="bold")
plt.axis('off')
plt.show()

# Pivot tables
jobs_groupby_City_title = jobs.groupby(['City','title'])['title'].count()
jobs_groupby_City_title.loc['New York']

title_city = jobs.pivot_table(index=['City','title'],values='Minimum_years_experience',aggfunc='median')
title_city.loc['New York']

# Takes in a string on text, remove all puntuation, stopwords, return the cleaned text as a list of words
stop = set (stopwords.words('english'))
def object_to_list (text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
object_to_list(Data_Science_Salary_Wage['content'])
X=Data_Science_Salary_Wage['content']
Y=Data_Science_Salary_Wage['title']
from sklearn.feature_extraction.text import CountVectorizer
X = CountVectorizer(analyzer = object_to_list).fit_transform(X)
print(X[6].split)

