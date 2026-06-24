import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 rows of dataset:")
print(df.head())



# Check Dataset Information

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())



# Handle Missing Values


# Fill missing Age values with median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Cabin values with "Unknown"
df["Cabin"] = df["Cabin"].fillna("Unknown")

#checking values after filling data

print("\nafter filling missing value")
print(df.isnull().sum())

#  Convert Text Data into Numbers


encoder = LabelEncoder()

# Convert male/female into numbers
# female = 0
# male = 1
df["Sex"] = encoder.fit_transform(df["Sex"])

# Convert S, C, Q into numbers
df["Embarked"] = encoder.fit_transform(df["Embarked"])



# Select Features and Target


# Features used for prediction
X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]

# Target column
y = df["Survived"]


# Split Data into Train and Test


# 80% data for training
# 20% data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)



# Create Machine Learning Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)



#  Train the Model


model.fit(X_train, y_train)



# Make Predictions


predictions = model.predict(X_test)


#  Calculate Accuracy


accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))



#  Predict Survival of a New Passenger


new_passenger = pd.DataFrame({
    "Pclass": [3],
    "Sex": [1],        
    "Age": [22],
    "SibSp": [1],
    "Parch": [0],
    "Fare": [7.25],
    "Embarked": [2]    
})

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("\nPrediction: Passenger would SURVIVE")
else:
    print("\nPrediction: Passenger would NOT survive")