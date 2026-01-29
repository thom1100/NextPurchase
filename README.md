# NextPurchase
Business Case to recommend new articles

# Install the datasets in your data folder

You need those files in data/raw :
- clients.csv
- products.csv
- stocks.csv
- stores.csv
- transactions.csv

Once you run the EDA, you will get in data/transformed :
- tx.csv

To get the other data/transformed :

- Go to scripts using ```cd scripts```
- in the terminal :
```python dataset_builder.py```
```python feature_engineering.py```

You should see the following csv appear :
- df_family_2.csv
- df_model_family_2.csv
- df_model_final_family_2.csv

The last one will be used in the notebook xgb_model.ipynb

# Activate your venv

Use the following command to activate the venv:
- ```source venv/bin/activate```
- ```pip install -r requirements.txt```



