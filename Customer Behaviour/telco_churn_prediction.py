import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Read the data
    df = pd.read_csv(file_path)
    
    # Convert TotalCharges to numeric, handling any spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Separate binary and categorical columns
    binary_columns = ['Churn', 'PhoneService', 'PaperlessBilling', 'Partner', 'Dependents']
    other_categorical = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod', 'gender',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Create a copy of the dataframe
    df_encoded = df.copy()
    
    # Manual mapping for binary columns (Yes/No)
    binary_map = {'No': 0, 'Yes': 1}
    for col in binary_columns:
        df_encoded[col] = df_encoded[col].map(binary_map)
    
    # Use LabelEncoder for other categorical columns
    le = LabelEncoder()
    for col in other_categorical:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        # Print mapping for debugging
        unique_values = df[col].unique()
        encoded_values = le.transform(unique_values)
        mapping = dict(zip(unique_values, encoded_values))
        print(f"\nMapping for {col}:")
        for original, encoded in mapping.items():
            print(f"{original} -> {encoded}")
    
    # Drop customerID as it's not relevant for prediction
    df_encoded = df_encoded.drop('customerID', axis=1)
    
    # Separate features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns

# Build TensorFlow model
def build_tensorflow_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Train TensorFlow model
def train_tensorflow_model(X_train, y_train, X_val, y_val):
    model = build_tensorflow_model((X_train.shape[1],))
    
    history = model.fit(X_train, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_data=(X_val, y_val),
                       verbose=1)
    return model, history

# Train XGBoost model
def train_xgboost_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        random_state=42
    )
    
    model.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             verbose=True)
    return model

# Combine predictions
def ensemble_predictions(tf_model, xgb_model, X_test):
    tf_pred = tf_model.predict(X_test)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    
    # Average the predictions
    ensemble_pred = (tf_pred.flatten() + xgb_pred) / 2
    return (ensemble_pred > 0.5).astype(int)

# Plot feature importance
def plot_feature_importance(xgb_model, feature_names):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.head(10))
    plt.title('Top 10 Most Important Features for Churn Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data('Odyssey 1  Data Set - Telco Data.csv')
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train TensorFlow model
    print("Training TensorFlow model...")
    tf_model, history = train_tensorflow_model(X_train, y_train, X_val, y_val)
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Make predictions using ensemble
    print("\nMaking ensemble predictions...")
    y_pred = ensemble_predictions(tf_model, xgb_model, X_test)
    
    # Print results
    print("\nEnsemble Model Performance:")
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    plot_feature_importance(xgb_model, feature_names)
    print("\nFeature importance plot has been saved as 'feature_importance.png'")

if __name__ == "__main__":
    main()
