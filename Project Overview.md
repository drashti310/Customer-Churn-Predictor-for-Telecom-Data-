## Project Objective:

• Implement an end-to-end machine learning model using Telecom data to develop a product which predicts how likely a customer will churn. <br>
• Using this product, a telecommunication company can identify those who are likely to lapse/churn and create strategies
to ensure customer retention (via customer relationship or marketing initiatives).

### Data:

The ‘Telco Customer Churn’ data containing information about a fictional telecom company that provides home phone and Internet services has been sourced from two publicly available datasets. The primary dataset published by IBM contains information pertaining to 7,043 unique customers in California in Q3 of a specific year. Since we needed more data to create a robust machine learning model, we combined it with a secondary dataset available on Kaggle which contains records pertaining to another set of 7,043 customers. The two datasets indicate the customers who have left, stayed, or signed up for their service. Various additional attributes are provided for each customer, like churn score, satisfaction score, and customer lifetime value (CLTV) index. The final dataset in scope comprises of 14,086 customer records with 33 variables.

![Data Description](https://user-images.githubusercontent.com/99310137/206799955-b5cbbfdb-eea3-4885-99af-373857dbabe1.jpg)

### Methodology:

First we train a robust machine learning model using the available data; that will accurately predict customers who are about to churn, which in turn enables the telecom stakeholders to make marketing or business expansion strategies backed by data science. 

![image](https://user-images.githubusercontent.com/99310137/206801034-4d88debc-d60f-4f0c-b930-6280ff3203ce.png)

#### Performance of all the models:


![Model performance](https://user-images.githubusercontent.com/99310137/206801306-f3ba9cbb-680f-47bf-b34f-d905d77c17dd.jpg)


### Model Deployment:

The selected model will now serve as the base framework for the web application built using Python library ‘Streamlit’. Streamlit converts data scripts into shareable web apps. Python library ‘Pickle’ is used to save and export the performing machine learning model to a Streamlit application which offers an inference to interact with the model. This step is what gives our model a real purpose, where the UI fetches user input, scores it against the model, and then shows the predicted results in real time.


### Product Demo:

![Customer will churn Demo](https://user-images.githubusercontent.com/99310137/206801431-b4e092a3-f536-4538-85fe-7b4660955764.gif)


![Customer won't churn Demo](https://user-images.githubusercontent.com/99310137/206801468-30e4541c-4335-4efc-8bde-412d29b1ff53.gif)
