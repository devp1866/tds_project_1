import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np


# Load the users.csv and repositories.csv files into DataFrames
users_df = pd.read_csv('users.csv')
repos_df = pd.read_csv('repositories.csv')

# Convert 'created_at' column to datetime format in users_df
users_df['created_at'] = pd.to_datetime(users_df['created_at'], errors='coerce')


# Question 1:  Who are the top 5 users in Berlin with the highest number of followers? List their login in order, comma-separated.
top_followed_users = users_df.nlargest(5, 'followers')['login']
top_followed_users_list = ', '.join(top_followed_users)
print("Top 5 users by followers:", top_followed_users_list)

# Question 2: Who are the 5 earliest registered GitHub users in Berlin? List their login in ascending order of created_at, comma-separated.
earliest_registered_users = users_df.nsmallest(5, 'created_at')['login']
earliest_registered_users_list = ', '.join(earliest_registered_users)
print("5 earliest registered users:", earliest_registered_users_list)

# Question 3: What are the 3 most popular license among these users? Ignore missing licenses. List the license_name in order, comma-separated.
# Filter out missing licenses
licenses = repos_df['license_name'].dropna()
top_licenses = licenses.value_counts().nlargest(3).index.tolist()
top_licenses_list = ', '.join(top_licenses)
print("3 most popular licenses:", top_licenses_list)

# Question 4: Which company do the majority of these developers work at?
users_df['company'] = users_df['company'].str.strip().str.lstrip('@').str.upper()
top_company = users_df['company'].mode().values[0]
print("Company with the majority of developers:", top_company)

# Question 5: Which programming language is most popular among these users?
languages = repos_df['language'].dropna()
top_language = languages.value_counts().idxmax()
print("Most popular programming language:", top_language)






# Convert 'created_at' column to datetime format in users_df
users_df['created_at'] = pd.to_datetime(users_df['created_at'], errors='coerce')

# Question 6: Which programming language is the second most popular among users who joined on or after 1 Jan 2020?
users_after_2020 = users_df[users_df['created_at'] > '2020-01-01']
# Get a list of their logins
logins_after_2020 = users_after_2020['login']
# Filter repositories to only those belonging to users who joined after 2020
repos_after_2020 = repos_df[repos_df['login'].isin(logins_after_2020)]
# Find the second most popular language
second_popular_language = repos_after_2020['language'].value_counts().index[1]
print("Second most popular language among users who joined after 2020:", second_popular_language)


# Question 7: Which language has the highest average number of stars per repository?
avg_stars_per_language = repos_df.groupby('language')['stargazers_count'].mean()
top_language_by_stars = avg_stars_per_language.idxmax()
print("Language with the highest average stars per repository:", top_language_by_stars)


# Question 8: Let's define leader_strength as followers / (1 + following). Who are the top 5 in terms of leader_strength? List their login in order, comma-separated.
users_df['leader_strength'] = users_df['followers'] / (1 + users_df['following'])
# Get the top 5 users by leader_strength
top_leader_strength_users = users_df.nlargest(5, 'leader_strength')['login']
top_leader_strength_users_list = ', '.join(top_leader_strength_users)
print("Top 5 users by leader_strength:", top_leader_strength_users_list)




# Question 9: What is the correlation between the number of followers and the number of public repositories among users in Berlin?
# Correlation between followers and repos (to 3 decimal places, e.g. 0.123 or -0.123)
followers_repos_corr, _ = pearsonr(users_df['followers'], users_df['public_repos'])
print("Correlation between followers and public repositories:", round(followers_repos_corr, 3))


# Question 10: Does creating more repos help users get more followers? Using regression, estimate how many additional followers a user gets per additional public repository.
# Regression slope of followers on repos (to 3 decimal places, e.g. 0.123 or -0.123)
X = users_df[['public_repos']].values
y = users_df['followers'].values
reg = LinearRegression().fit(X, y)
followers_repos_slope = reg.coef_[0]
print("Regression slope of followers on public repositories:", round(followers_repos_slope, 3))


# Question 11: Do people typically enable projects and wikis together? What is the correlation between a repo having projects enabled and having wiki enabled?
# Correlation between projects and wiki enabled (to 3 decimal places, e.g. 0.123 or -0.123)
repos_df['has_projects'] = repos_df['has_projects'].astype(int)
repos_df['has_wiki'] = repos_df['has_wiki'].astype(int)
projects_wiki_corr, _ = pearsonr(repos_df['has_projects'], repos_df['has_wiki'])
print("Correlation between projects and wiki enabled:", round(projects_wiki_corr, 3))


# Question 12:  Do hireable users follow more people than those who are not hireable?
# Average of following per user for hireable=true minus the average following for the rest (to 3 decimal places, e.g. 12.345 or -12.345)
users_df['hireable'] = users_df['hireable'].fillna(False).astype(bool)
# Calculate average following for hireable and non-hireable users
avg_following_hireable = users_df[users_df['hireable'] == True]['following'].mean()
avg_following_non_hireable = users_df[users_df['hireable'] == False]['following'].mean()
following_difference = avg_following_hireable - avg_following_non_hireable
print("Difference in average following between hireable and non-hireable users:", round(following_difference, 3))





# Question 13: Some developers write long bios. Does that help them get more followers? What's the impact of the length of their bio (in Unicode words, split by whitespace) with followers? (Ignore people without bios)
# Regression slope of followers on bio word count (to 3 decimal places, e.g. 12.345 or -12.345)
users_with_bio = users_df.dropna(subset=['bio']).copy()
# Calculate word count for each bio
users_with_bio['bio_word_count'] = users_with_bio['bio'].str.split().apply(len)
# Perform linear regression with bio_word_count as the independent variable and followers as the dependent variable
X_bio = users_with_bio[['bio_word_count']].values
y_followers = users_with_bio['followers'].values
reg_bio = LinearRegression().fit(X_bio, y_followers)
bio_followers_slope = reg_bio.coef_[0]
print("Regression slope of followers on bio word count:", round(bio_followers_slope, 3))


# Question 14: Who created the most repositories on weekends (UTC)? List the top 5 users' login in order, comma-separated
repos_df['created_at'] = pd.to_datetime(repos_df['created_at'], errors='coerce')
# Filter for repos created on weekends (Saturday=5, Sunday=6)
weekend_repos = repos_df[repos_df['created_at'].dt.weekday >= 5]
# Count the number of weekend repos per user and find the top 5
top_weekend_creators = weekend_repos['login'].value_counts().nlargest(5).index.tolist()
top_weekend_creators_list = ', '.join(top_weekend_creators)
print("Top 5 users by repos created on weekends:", top_weekend_creators_list)


# Question 15: Do people who are hireable share their email addresses more often?
# [fraction of users with email when hireable=true] minus [fraction of users with email for the rest] (to 3 decimal places, e.g. 0.123 or -0.123)
users_df['hireable'] = users_df['hireable'].fillna(False).astype(bool)
# Calculate fractions of users with email for hireable and non-hireable users
hireable_with_email = users_df[users_df['hireable'] & users_df['email'].notna()].shape[0] / users_df[users_df['hireable']].shape[0]
non_hireable_with_email = users_df[~users_df['hireable'] & users_df['email'].notna()].shape[0] / users_df[~users_df['hireable']].shape[0]
email_share_diff = hireable_with_email - non_hireable_with_email
print("Difference in email sharing between hireable and non-hireable users:", round(email_share_diff, 3))


# Question 16: Let's assume that the last word in a user's name is their surname (ignore missing names, trim and split by whitespace.) What's the most common surname? (If there's a tie, list them all, comma-separated, alphabetically)
# Most common surname(s)
users_with_names = users_df.dropna(subset=['name']).copy()
users_with_names['surname'] = users_with_names['name'].str.strip().str.split().str[-1]
# Find the most common surname(s)
surname_counts = users_with_names['surname'].value_counts()
max_surname_count = surname_counts.max()
most_common_surnames = surname_counts[surname_counts == max_surname_count].index.tolist()
most_common_surnames.sort()  # Sort alphabetically if there's a tie
most_common_surnames_list = ', '.join(most_common_surnames)
print("Most common surname(s):", most_common_surnames_list)
