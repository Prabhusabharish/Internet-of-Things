import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Set up the Streamlit app

#  ------------------------------------------ Streamlit Part ---------------------------------------------------------

st.set_page_config(layout= "wide")

st.markdown(
    f""" <style>.stApp {{
                    background:url("https://wallpapers.com/images/high/dark-purple-and-black-plain-75znhgkjjxu552fr.webp");
                    background-size:cover}}
                 </style>""",
    unsafe_allow_html=True
)
st.markdown(
    """
    <h1 style="text-align: center; color: orange;">**Advanced IoT Greenhouse and Traditional Greenhouse Plant Data**</h1>
    """,
    unsafe_allow_html=True
)

#st.title(":orange[**Advanced IoT Greenhouse and Traditional Greenhouse Plant Data**]")
#st.write("Upload a dataset and perform EDA, train models, and evaluate their performance.")

# Sidebar for navigation
page = st.sidebar.radio("Select Page", ["Home", "Data", "EDA", "Model Training", "Add/Delete Data"])

# Function to load and preprocess data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    df['Random'] = df['Random'].astype('category').cat.codes
    df['Class'] = df['Class'].astype('category').cat.codes
    return df

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if page == "Home":
        st.write("### Home")
        st.write("Explore, analyze, and model plant data from advanced IoT and traditional greenhouses:")
        st.write("- **EDA:** Visualize and analyze your dataset with histograms, box plots, and correlation matrices.")
        st.write("- **Model Training:** Train and evaluate machine learning models like Random Forest, SVM, and more to predict plant classifications.")
        st.write("- **Add/Delete Data:** Easily manage dataset entries with options to add new data or remove existing records.")
        
        st.write("Upload your CSV file to begin uncovering insights into plant growth and greenhouse operations!")

        
    if page == "Data":
        #st.write("### Data Preprocessing")
        
        st.write("### Dataset")
        st.write(df.head())
        
        st.write("### Data Preprocessing")
        st.write("Missing Values:\n", df.isnull().sum())
        st.write("Dataset Dimensions:", df.shape)
        st.write("Column Names:", df.columns)
        st.write("\nData Types:\n", df.dtypes)
        st.write("\nSummary Statistics:\n", df.describe())
        
     
    if page == "EDA":
        st.write("### Exploratory Data Analysis")
        
        # Histograms for numeric columns
        st.write("Histograms for numeric columns")
        hist_fig, ax = plt.subplots(figsize=(12, 10))  
        df.hist(ax=ax)  
        plt.tight_layout()
        st.pyplot(hist_fig)
        
        # Box plots for numeric columns
        st.write("Box plots for numeric columns")
        box_fig = plt.figure(figsize=(12, 8))
        sns.boxplot(data=df)
        plt.xticks(rotation=45)
        st.pyplot(box_fig)
        
        # Count plots for categorical columns
        st.write("Count plots for categorical columns")
        count_fig = plt.figure(figsize=(10, 6))
        sns.countplot(x='Class', data=df)
        plt.title('Class Distribution')
        st.pyplot(count_fig)
        
        # Correlation matrix
        st.write("Correlation Matrix")
        correlation_matrix = df.corr()
        corr_fig = plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        st.pyplot(corr_fig)
        
        # Pair plot with selected columns
        st.write("Pair Plot")
        df_corrected = df.rename(columns={' Average  of chlorophyll in the plant (ACHP)': 'ACHP',
                                          ' Plant height rate (PHR)': 'PHR',
                                          'Average leaf area of the plant (ALAP)': 'ALAP'})
        pair_fig = sns.pairplot(df_corrected[['ACHP', 'PHR', 'ALAP', 'Class']], hue='Class', diag_kind='hist')
        st.pyplot(pair_fig)

        # Box plot for a specific column
        st.write("Box plot of ACHP by Class")
        box_fig2 = plt.figure(figsize=(12, 8))
        sns.boxplot(x='Class', y='ACHP', data=df_corrected)
        plt.title('Box plot of ACHP by Class')
        st.pyplot(box_fig2)

    if page == "Model Training":
        st.write("### Model Training and Evaluation")
        
        # Split the Data
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training and Evaluation
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Support Vector Machine': SVC(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        results = []
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=1)
            st.write(f'### {model_name} Results')
            st.write(f'Accuracy: {accuracy}')
            st.write(f'Classification Report:\n{report}')
            results.append({
                'model': model_name,
                'accuracy': accuracy,
                'report': report
            })

        results_df = pd.DataFrame(results)
        st.write("### Model Comparison")
        st.write(results_df)
        
        # Save the best model (example with the last trained model)
        joblib.dump(model, 'trained_model.pkl')
        st.write("Best model saved as 'trained_model.pkl'")
        
        # Assuming df contains your dataset
        df_sample = pd.DataFrame({
            'Random': ['R1', 'R1', 'R2', 'R1', 'R3'],
            'Average of chlorophyll in the plant (ACHP)': [34.533468, 34.489028, 33.100405, 34.498319, 36.297008],
            'Plant height rate (PHR)': [54.566983, 54.567692, 67.067344, 54.559049, 45.588894],
            'Average wet weight of the growth vegetative (AWWGV)': [1.147449, 1.149530, 1.104647, 1.137759, 1.363205],
            'Average leaf area of the plant (ALAP)': [1284.229549, 1284.247744, 1009.208996, 1284.227623, 981.470310],
            'Average number of plant leaves (ANPL)': [4.999713, 5.024259, 5.007652, 4.991501, 4.003682],
            'Average root diameter (ARD)': [16.274918, 16.269452, 15.980760, 16.276710, 16.979894],
            'Average dry weight of the root (ADWR)': [1.706810, 1.700930, 1.185391, 1.716396, 0.777428],
            'Percentage of dry matter for vegetative growth (PDMVG)': [18.399982, 18.398288, 19.398789, 18.413613, 31.423772],
            'Average root length (ARL)': [19.739037, 19.758836, 20.840822, 19.736098, 17.331894],
            'Average wet weight of the root (AWWR)': [2.949240, 2.943137, 2.861635, 2.946784, 2.766242],
            'Average dry weight of vegetative plants (ADWV)': [0.209251, 0.216154, 0.200113, 0.223092, 0.424172],
            'Percentage of dry matter for root growth (PDMRG)': [57.633906, 57.633697, 41.289875, 57.645661, 27.898619],
            'Class': ['SA', 'SA', 'SA', 'SA', 'SA']
        })

        # Simulate or generate 'Predicted_Class' based on some criteria
        np.random.seed(0)
        df_sample['Predicted_Class'] = np.random.choice(['SA', 'SB', 'SC'], size=len(df_sample))

        # Assuming 'Class' and 'Predicted_Class' are columns in your dataset
        y_true = df_sample['Class'].values  
        y_pred = df_sample['Predicted_Class'].values  

        # Generate confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.write(cm)

    if page == "Add/Delete Data":
        st.write("### Add/Delete Data")
        
        # Functionality to add data
        st.write("### Add Data")
        if st.checkbox("Show Add Data Form"):
            new_data = {}
            for col in df.columns:
                if col != 'Class':
                    new_data
