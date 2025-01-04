import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Streamlit App Title
st.title("Loan Default Prediction")
st.sidebar.header("Model Configuration")

# Upload Dataset
st.sidebar.subheader("Upload Dataset")
file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if file is not None:
    data = pd.read_csv(file)

    # Remove loadID column if it exists
    if 'loanID' in data.columns:
        data.drop(columns=['loanID'], inplace=True)

    # Encode categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        label_encoders = {col: LabelEncoder() for col in categorical_cols}
        for col in categorical_cols:
            data[col] = label_encoders[col].fit_transform(data[col])

    st.write("### Data Preview")
    st.write(data.head())
    st.write(data.describe().T)

    # Feature and Target Selection
    target = st.sidebar.selectbox("Select Target Column", options=data.columns)
    features = st.sidebar.multiselect("Select Feature Columns", options=data.columns, default=[col for col in data.columns if col != target])

    if target and features:
        X = data[features]
        y = data[target]

        # Train-Test Split
        test_size = st.sidebar.slider("Test Size (Fraction)", 0.1, 0.5, 0.3)
        random_state = st.sidebar.number_input("Random State", min_value=0, max_value=1000, value=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Model Selection
        st.sidebar.subheader("Choose Model")
        model_choice = st.sidebar.selectbox(
            "Select Model",
            ("Logistic Regression", "Random Forest", "Gradient Boosting", "Support Vector Machine")
        )

        # Hyperparameter Configuration
        st.sidebar.subheader("Hyperparameter Configuration")

        if model_choice == "Logistic Regression":
            penalty = st.sidebar.selectbox("Penalty", ["l2", "l1"], index=0)
            C = st.sidebar.number_input("C (Regularization Strength)", min_value=0.01, max_value=10.0, value=1.0)
            model = LogisticRegression(penalty=penalty, C=C, solver='liblinear')

        elif model_choice == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

        elif model_choice == "Gradient Boosting":
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
            n_estimators = st.sidebar.slider("Number of Boosting Stages", 10, 500, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 50, 3)
            model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

        elif model_choice == "Support Vector Machine":
            kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
            C = st.sidebar.number_input("C (Regularization Strength)", min_value=0.01, max_value=10.0, value=1.0)
            model = SVC(kernel=kernel, C=C, probability=True)

        # Train Model
        if st.sidebar.button("Train Model"):
            st.write(f"### Training {model_choice}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
            st.write(f"### Accuracy: {accuracy:.2f}")

            if hasattr(model, "feature_importances_"):
                st.write("### Feature Importance")
                importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
                st.write(importance.sort_values(by="Importance", ascending=False))

else:
    st.write("Upload a CSV file to get started.")
