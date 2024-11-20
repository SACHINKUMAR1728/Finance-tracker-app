import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# # Download nltk resources
# nltk.download('stopwords')
# nltk.download('wordnet')

# Set up lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Helper functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def generate_recommendation(row):
    if row['category'] == 'Food' and row['amount'] > 300:
        return "Consider cooking at home more often to save money on food."
    elif row['category'] == 'Apparel' and row['amount'] > 150:
        return "Try limiting clothing purchases to reduce spending on apparel."
    elif row['category'] == 'Transportation' and row['amount'] > 100:
        return "Consider using public transport or carpooling to save on transportation costs."
    else:
        return "Good job staying within your budget!"

# Streamlit App
st.set_page_config(page_title="Budget Categorization & Recommendations", layout="wide")

# Header
st.title("💰 Budget Categorization & Recommendation System")
st.markdown("Analyze and optimize your expenses with intelligent categorization and budgeting.")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Model configurations
model_configs = {
    "Multinomial Naive Bayes": {"alpha": [0.1, 0.5, 1.0, 5.0]},
    "Complement Naive Bayes": {"alpha": [0.1, 0.5, 1.0, 5.0]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
    "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ['linear', 'rbf'], "gamma": ['scale', 'auto']},
}

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    with st.container():
        st.subheader("📊 Dataset Overview")
        st.write("Preview of the uploaded dataset:")
        st.dataframe(df.head(), use_container_width=True)

    # Preprocess data
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce', dayfirst=True)
    df['description'] = df['description'].fillna('').apply(preprocess_text)
    df['category_encoded'] = LabelEncoder().fit_transform(df['category'])

    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek

    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(df['description'])
    y = df['category_encoded']

    # Model Selection
    st.sidebar.subheader("Model Selection")
    model_type = st.sidebar.selectbox("Choose a Model", model_configs.keys())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # GridSearchCV
    st.sidebar.subheader("Hyperparameter Tuning")
    param_grid = model_configs[model_type]

    if model_type == "Multinomial Naive Bayes":
        grid_search = GridSearchCV(MultinomialNB(), param_grid, scoring='accuracy', cv=3)
    elif model_type == "Complement Naive Bayes":
        grid_search = GridSearchCV(ComplementNB(), param_grid, scoring='accuracy', cv=3)
    elif model_type == "Random Forest":
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='accuracy', cv=3)
    elif model_type == "Support Vector Machine":
        grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=3)

    # Train and evaluate the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Display Metrics
    st.subheader(f"🧠 {model_type} Results")
    st.write(f"Best Parameters: {grid_search.best_params_}")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.text("Classification Report:")
        st.text(report)

    # Visualize Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Recommendations
    df['recommendation'] = df.apply(generate_recommendation, axis=1)
    with st.expander("💡 Recommendations"):
        st.write(df[['date', 'category', 'amount', 'recommendation']])

    # Budget Visualization
    st.subheader("📈 Spending vs. Budget")
    budgets = {
        'Apparel': 300,
        'Beauty': 100,
        'Education': 500,
        'Food': 3000,
        'Gift': 300,
        'Household': 700,
        'Other': 1000,
        'Self-development': 1500,
        'Social Life': 2000,
        'Transportation': 5000
    }

    df['month_year'] = df['date'].dt.to_period('M')
    monthly_expense = df.groupby(['month_year', 'category']).amount.sum().unstack().fillna(0)

    st.bar_chart(monthly_expense)

    # Budget vs. Spending Graph
    for category in budgets.keys():
             # Ensure data is numeric and handle NaN values
             category_data = pd.to_numeric(monthly_expense[category], errors='coerce').fillna(0)
             # Skip plotting if all values are zero
             if category_data.sum() == 0:
                st.warning(f"No spending data available for category: {category}")
                continue
             # Create the plot
             fig, ax = plt.subplots(figsize=(10, 5))
             sns.lineplot(x=monthly_expense.index.astype(str), y=category_data, label="Spending", ax=ax)
             ax.axhline(y=budgets[category], color='r', linestyle="--", label="Budget")
             ax.set_title(f"{category} Spending vs. Budget")
             ax.set_xlabel("Month-Year")
             ax.set_ylabel("Amount Spent")
             ax.legend()
             ax.grid(True)

             # Display the plot in Streamlit
             st.pyplot(fig)
