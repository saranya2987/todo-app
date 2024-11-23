import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load the CSV data
data = pd.read_csv("todo_tasks_data.csv")

# Check if the data is loaded properly
print(data.head())

# Define the features and the target
X = data[['text', 'description', 'category']]  # Ensure X is a DataFrame
y = data['priority']  # Target is the 'priority' column

# Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('text_tfidf', TfidfVectorizer(), 'text'),  # Apply TFIDF to 'text' column
        ('desc_tfidf', TfidfVectorizer(), 'description'),  # Apply TFIDF to 'description' column
        ('category_ohe', OneHotEncoder(), ['category'])  # Ensure 'category' is passed as a list (2D)
    ], 
    remainder='passthrough'  # Ensure all other columns are passed through without transformation
)

# Create the pipeline: first preprocessing, then model training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing step
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])

# Ensure that the input features are correctly structured
print(f"X_train shape: {X_train.shape}")  # Check if this is (n_samples, n_features)
print(f"X_test shape: {X_test.shape}")    # Check the test data shape

# Train the model with training data
model.fit(X_train, y_train)

# Evaluate the model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "priority_model_with_categories.pkl")
print("Model trained and saved as priority_model_with_categories.pkl")
