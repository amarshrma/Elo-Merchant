# Elo-Merchant
Elo, is one of the largest payments brands in brazil, which wants to predict a loyalty score for each of its customer.
This is a kaggle competition which can be accessed by [this](https://www.kaggle.com/c/elo-merchant-category-recommendation) link.
This repository contains solutions related to the competition. Loyalty score which is predicted for each of its customer, indicates how useful are the services of Elo to that particular customer. By doing so, it can focus more on those loyal customers and also serve them better.

This is basically a regression problem, with loyalty_Score as its prediction. I have tried various models but what really worked for me is the [LightGBM](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) model. It is one of faster variants of Gradient Boosting.

I have also tried to pose this as a classification problem with classes being ['Loyal', 'Not Loyal'].

I have also hosted this solution on an amazon cloud with an UI where users can select various card id's and some features and based on that it will output a loyalty score.
