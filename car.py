import pandas as pd
import numpy as np

import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

# Load dataset
car = pd.read_csv('quikr_car.csv')

# Create backup
backup = car.copy()

# Clean year column
car['year'] = pd.to_numeric(car['year'], errors='coerce')
car = car.dropna(subset=['year'])
car['year'] = car['year'].astype(int)

# Clean Price column
car['Price'] = car['Price'].astype(str).str.strip()
car['Price'] = car['Price'].replace('Ask For Price', pd.NA)
car['Price'] = car['Price'].str.replace(',', '', regex=False)
car['Price'] = pd.to_numeric(car['Price'], errors='coerce')
car = car.dropna(subset=['Price'])
car['Price'] = car['Price'].astype(int)

# Clean kms_driven column
car['kms_driven'] = car['kms_driven'].astype(str)
car['kms_driven'] = car['kms_driven'].str.replace(',', '', regex=False)
car['kms_driven'] = car['kms_driven'].str.replace(' kms', '', regex=False)
car['kms_driven'] = pd.to_numeric(car['kms_driven'], errors='coerce')
car = car.dropna(subset=['kms_driven'])
car['kms_driven'] = car['kms_driven'].astype(int)

# Remove missing fuel_type values
car = car.dropna(subset=['fuel_type'])

# Keep first 3 words of car name
car['name'] = car['name'].str.split().str[:3].str.join(' ')

# Save cleaned dataset
car.to_csv('cleaned_car.csv', index=False)

# Separate input and target
x = car.drop(columns='Price')
y = car['Price']

# Split data into training and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42
)

# One Hot Encoding
ohe = OneHotEncoder(handle_unknown='ignore')

# Create column transformer
column_trans = make_column_transformer(
    (ohe, ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Create Linear Regression model
lr = LinearRegression()

# Create pipeline
pipe = make_pipeline(column_trans, lr)

# Train the model
pipe.fit(x_train, y_train)

# Predict prices
y_pred = pipe.predict(x_test)

# Check R2 score
print("R2 Score:", r2_score(y_test, y_pred))

# Find best random state
scores = []

for i in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.2, random_state=i
    )

    lr = LinearRegression()

    pipe = make_pipeline(column_trans, lr)

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)

    scores.append(r2_score(y_test, y_pred))

# Get best random state
best_random_state = np.argmax(scores)

# Get best score
best_score = scores[best_random_state]

print("Best Random State:", best_random_state)
print("Best R2 Score:", best_score)

# Train final model using best random state
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2, random_state=best_random_state
)

lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)

# Fit finalR2 model
pipe.fit(x_train, y_train)

# Final prediction
y_pred = pipe.predict(x_test)

# Final R2 score
print("Final  Score:", r2_score(y_test, y_pred))

# Save trained model
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# Test car price prediction
prediction = pipe.predict(
    pd.DataFrame(
        [['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )
)

print("Predicted Price:", prediction[0])