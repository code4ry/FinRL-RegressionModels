# Logistic Regression Model Milestones

**What is Logistic Regression**

Logistic Regression is the type of statistical model that estimates the probability of
an event occuring, based on a given data and a set of independent variables, that produce
the dependent variable outcome bounded by 0 and 1. When using Logistic Regression, the 
logit transformation is applied on the odds--that is, the probability of success divided by
the probability of failure. This is commonly known as log odds, or natural logarithm of odds. 
This is represented through two formulas:

Logit(pi) = 1/(1+ exp(-pi))

ln(pi/(1-pi)) = Beta_0 + Beta_1*X_1 + … + B_k*K_k

**Milestone 1:**

02/02: I assembled and finalized the people that are in my project group for RCOS. Then, during the RCOS small groups, we completed our project proposal, discussing what our milestones for the project are and assigning different regression models to each team member to work on. In addition, we even found some really good stock market datasets that we can use and standardize throughout the project for the team to use when building out our models. Finally, we were getting our developing environment ready to go to start building our initial models soon. 

02/09: We finalized the datasets we will be using for this week, began researching what our regression method and logic is, and began coding the visualization of our data. In addition, we also learned about the duties of the scrum and built paper airplanes. We went through a simulation of what it's like to plan, design and build, and then reflect and improve on our paper airplane. I think this was very insightful since there will be a similar dynamic within our RCOS project groups.

**Milestone 2:**

02/16: Our group finished visualizing the datasets for our groups and have updated the README/documentation of our project. As for me, I was figuring out how to download and get familiar with the Python libraries such as NumPy, Pandas, and Matplotlib. In addition, I am getting used to using Google CoLaboratory notebooks to code our project. Finally, I attended a Quantum Computing Club workshop during Tuesday's small breakout groups.

02/23: I got my libraries to comply in my development environment. In addition, I visualized the data and started to train, test, and split the data to prepare for creating our initial model. I did more research online on to how I should best train test and split the data, as well as collaborate with my group to find the best way to train, test, and split the data.

**Milestone 3:**

03/01: This week, I was able to train test and split my data for building the initial model. However, I am experimenting with pandas Time Series, trying to understand and implement the advantages of using Time Series to train test and split my data. Otherwise, the preparation of my data will need more time.

03/22: This week, I have finished and evaluated the errors in the initial model. I found that the model is experiencing data overfitting, in addition, I have a mean squared error that isn't too desirable. I am researching ways to improve the initial model to make it more accurate, and to hopefully reduce overfitting of data.

03/29: This week, I found a way to best visualize my predicted data along with my actual data from the dataset to show how accurate the prediction model is compared to the actual movement of the actual data. In addition, I discovered cross-validation. Using this, I was able to split my data set into five "folds" to create five different iterations of the train test splitting my data, using 4 folds for training, and 1 fold for testing. Then, gathering all the mean squared errors, I returned an average cross-validation score, which improved the accuracy of the model. 

04/05: This week, I have been researching more ways to improve the model by researching regularization. In addition, I have been looking for more ways to represent my predicted data that my machine learning model spews out. But, I was able to get my first data graph of my predicted data from my model to graph well.

04/12: This week, I attempted to find different methods of graphing the binary data that I was able to predict using my model. It was quite unsuccessful since box-plots and other common methods to graph binary data doesn't work. But, I am practically finished with my model and I was coordinating my model results with my other teammates.