# Liam Ralph Technical Exercise

## Question 1
Results are in ./data/output.csv under the "InPlayProbability" Column
The model predicts a 27.23% chance of a ball being in play, which compares well to the training data's 27.17% chance.

## Question 2
The first decision was made after initially exploring the data set. I found that a few of the pitch entries had NULL data in a few columns, which brings me to the second decision. This decision was what type of model I would use. I decided on XGBoost's implementation of a Gradient Boosted Classifier, since the data is not noisy and contains values within the same bounds as the training data (No extrapolation needed), so XGBoost makes sense. Additionally, although we want probabilities, it is a classification problem at hand, so a classifier makes more sense compared to a regressor. XGBoost also has imputation built in by default, dealing with the NULL values for us by filling in a value based on similar pitches. The choice of Random Forest vs Gradient Boosted Trees was made after hyperparameter tuning, and picking the highest accuracy, but the two have the same results in this problem space. The next choice was hyperparameter tuning, which was completed using grid search cross validation on a limited subset of parameters that I chose (It is my understanding that default parameters generally work well enough, and for a problem like this, improvement is more easily made by adding/changing features, plus I wasn't going to wait hours for Grid Search to complete).

## Question 3
If I could only show them one visual I'd chose this table:
![Image](https://i.imgur.com/ekGYVa7.png)

This matrix shows the correlation between feature inputs, as well as the output.
What I would tell the pitcher:
Using this table we can see how each factor of your pitches are related, as well as how it affects the outcome showing you that the most important thing to note is that in play probability is most affected by Vertical Break, then Spin Rate, then Horizontal Break, and lastly Velocity. As for the best area to improve in to decrease balls in play, I would suggest working on Spin Rate, while it doesn't have the biggest impact on outcome, it plays a role in how much your pitches break Horizontally and Vertically, leading to less balls in play.

## Question 4
For further development of the model, I would like to look at how easy it is for the pitcher to change the inputs of the model. For example, a model might say increasing Velocity is the best way to decrease Balls in Play Rate, but, Velocity might be the hardest skill to increase for the given pitcher, increasing Spin Rate could have lower impact, but the pitcher might have an easier time increasing it. An additional idea would be looking at lefty/righty splits.

## Question 5
Code is included in the repository.
