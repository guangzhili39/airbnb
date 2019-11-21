There is a file listings.csv which provides data on around 50,000 AirBnB listings in New York City.  
Given only this data, we want to create a model to predict how much you can charge for new listings 
while keeping vacancy down.  
In this project, we build a set of prediction models, including linear regression, lasso regualization,
svm regression and knn regression prediction models. Then we use votingregressor to ensemble these models 
together and use the average value to recommend the final price.  
This project goes one step further: we build the ensemble model using existing data first,
then generate the trained model. The model could be placed in any server.
The server application could load this model and generate a web application.
Then end user could open the application on a web page, input the property information, the application server will
apply the corresponding trained model to recommend the right price for the property under certain vacancy threshold,
which is given by the user.
How to run them?
1) to build the model, in cmd terminal, run python airbnbmodel_final.py
2) to run application server, in cmd terminal, run python serverdemo.py
3) to run application client, in google chrome web page, type localhost:33333

