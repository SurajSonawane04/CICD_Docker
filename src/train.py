import pandas
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the dataset
df = pd.read_csv("data/students.csv")

# Feature
x = df[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
]

# Target
y = df["result"]

# convert pass and fail  1 / 0
encoder = LabelEncoder
y = encoder.fit_transform(y)

# split the dataset
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# train 
model.fit(x_train,y_train)

# prediction
y_pred = model.predict(x_test)

# model accurancy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# save model
joblib.dump(
    {
        "model": model,
        "encoder": encoder
    },
    "model/model.pkl"
)

print("Model saved successfully!")