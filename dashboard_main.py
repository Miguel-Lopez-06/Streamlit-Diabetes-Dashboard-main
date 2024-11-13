# -------------------------

# Library Imports

# Streamlit
import streamlit as st

# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.utils import resample

# Importing Models
import joblib

# Images
from PIL import Image

# -------------------------

# Page configuration
st.set_page_config(
    page_title="Diabetes", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# -------------------------

# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('Diabetes Dashboard')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training two classification models using the Diabetes dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1UWI3jhR9aK2h7xXVTs4qAiTrXt3dyOSu?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Miguel-Lopez-06/Streamlit-Diabetes-Dashboard-main)")
    st.markdown("by: [`AM7-Group 6`](https://github.com/Miguel-Lopez-06/Streamlit-Diabetes-Dashboard-main)")
    st.subheader('Members:')
    st.write('Lu, Angel Michael   ')
    st.write('Libres, Francis Joseph   ')
    st.write('Lopez, John Finees Miguel   ')
    st.write('Molina, Juan Miguel   ')
    st.write('Macaraeg, Vincent Angelo   ')


# -------------------------

# Data

# Load data
diabetes_df = pd.read_csv("data/diabetes.csv")
diabetes_df['Outcome'] = diabetes_df['Outcome'].map({1: 'Diabetes', 0: 'No Diabetes'})

# -------------------------

# Importing models

dt_classifier = joblib.load('assets/models/decision_tree_model.joblib')
log_reg = joblib.load('assets/models/logistic_regression_model.joblib')


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
outcome_list = ['Diabetes', 'No Diabetes']

# -------------------------

# Plots

def feature_importance_plot(feature_importance_df, width, height, key):
    # Generate a bar plot for feature importances
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h'  # Horizontal bar plot
    )

    # Adjust the height and width
    feature_importance_fig.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot_{key}")


# -------------------------

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown(""" 

    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to classify Diabetes outcome from the Diabetes dataset (Diabetes and No Diabetes) using **Decision Tree Classifier** and **Logistic Regression**.

    #### Pages
    1. `Dataset` - Brief description of the Diabetes dataset used in this dashboard. 
    2. `EDA` - Exploratory Data Analysis of the Diabetes dataset. Highlighting the distribution of Diabetes and the relationship between the features. Includes graphs such as Pie Chart, Histogram, Box Plot, and Scatter Plot.
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the outcome column and splitting the dataset into training and testing sets.
    4. `Machine Learning` - Training two supervised classification models: Decision Tree Classifier and Logistic Regression. Includes model evaluation, Feature Importance, Confusion Matrix and ROC Curve.
    5. `Prediction` - Prediction page where users can input values to predict the Diabetes outcome using the trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.


    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""

    The **Diabetes dataset** is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.  

    Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. This dataset is commonly used to test classification techniques like support vector machines. The same dataset that is used for this data science activity was uploaded to Kaggle by the user named **Mehmet Akturk**.

    **Content**  
    The dataset has 768 rows containing 9 primary attributes that are related to medical information for predicting diabetes. The columns are as follows: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome (0 for No Diabetes, 1 for Diabetes).

    `Link:` https://www.kaggle.com/datasets/mathchi/diabetes-data-set            
                
    """)

    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(diabetes_df, use_container_width=True, hide_index=True)

    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(diabetes_df.describe(), use_container_width=True)

    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset. The dataset comprises 768 entries, each with 9 attributes that capture various health and medical metrics relevant to diabetes risk. On average, individuals have had approximately 3.85 pregnancies, with a significant range from 0 to 17, reflecting a diverse population in terms of reproductive history. Glucose levels average around 120.89, though there is moderate variability, with values ranging from 0 to 199. The median glucose level is 117, suggesting some extreme values on both ends, possibly indicating outliers or data errors for cases with a glucose level of 0.  

    Blood pressure levels also show variability, with an average of 69.11 and a standard deviation of 19.36. Blood pressure readings range from 0 to 122, with a median of 72. The presence of zero values may imply missing or incorrect data. Skin thickness has a mean of 20.54, a standard deviation of 15.95, and a range extending up to 99. This measure includes some zero values, likely indicating missing data, while the median of 23 suggests a slight skew toward higher values. Insulin levels vary greatly, with an average of 79.80 but a high standard deviation of 115.24, spanning from 0 to 846. This wide spread suggests significant variation in insulin levels among individuals, potentially influenced by factors like diabetes management or data quality.  

    The dataset’s BMI (Body Mass Index) averages 31.99, with values ranging from 0 to 67.1, and a median of 32. While this indicates a distribution around typical BMI values, the presence of zero values may indicate missing entries. The Diabetes Pedigree Function, reflecting genetic predisposition, has a mean of 0.4719 and ranges from 0.078 to 2.42, with a median of 0.3725. This suggests that a subset of individuals has a notably higher genetic risk for diabetes. Age-wise, the dataset’s participants average 33.24 years, with a standard deviation of 11.76, indicating a relatively young to middle-aged group, and ages span from 21 to 81 years, with a median age of 29.
                
    """)


# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    col = st.columns((5, 5, 5), gap='medium')

    with col[0]:

        with st.expander('Legend', expanded=True):
            st.write('''
                - Data: [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).
                - :orange[**Pie Chart**]: Distribution of Outcome.
                - :orange[**Histogram**]: Age Distribution.
                - :orange[**Box Plot**]: Glucose Levels by Outcome.
                - :orange[**Scatter Plot**]: BMI vs Age (Color-coded by Outcome).
                
                ''')

    st.header("Distribution of Outcome")        
    dt_tree_image = Image.open('assets/eda/PieChart.png')
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths if needed
    with col1:
        st.write("")  # Placeholder for left column
    with col2:
        st.image(dt_tree_image, caption='Distribution of Outcome', use_column_width=True)  # Centered image
    with col3:
        st.write("")

    st.markdown("""
                
    Based on the results we can see that there's a balanced distribution of the outcome of Diabetes. With this in mind, we don't need to perform any pre-processing techniques anymore to balance the classes since it's already balanced.  
                
    """)

    st.header("Age Distribution")
    dt_tree_image = Image.open('assets/eda/PieChart.png')
    st.image(dt_tree_image, caption='Age Distribution')

    st.markdown("""
                
    The plot shows that the majority of individuals are between 20 and 40 years old, with a significant decrease in frequency as age increases beyond 50. This distribution can indicate that the dataset has a younger population focus. Understanding age distribution is important because age can be a significant factor in diabetes risk, especially as age increases.  
                
    """)

    st.header("Glucose Levels by Outcome")
    dt_tree_image = Image.open('assets/eda/PieChart.png')
    st.image(dt_tree_image, caption='Glucose Levels by Outcome')

    st.markdown("""
                
    The plot reveals that individuals with diabetes (Outcome = 1) generally have higher glucose levels than those without diabetes (Outcome = 0). There are also more outliers in the group without diabetes, with some individuals having very low glucose levels. This clear difference suggests that glucose levels are an important factor in predicting diabetes outcomes, as higher glucose levels are commonly associated with diabetes.  
                
    """)

    st.header("BMI vs Age (Color-coded by Outcome)")
    dt_tree_image = Image.open('assets/eda/PieChart.png')
    st.image(dt_tree_image, caption='BMI vs Age (Color-coded by Outcome)')

    st.markdown("""
                
    In the plot, red and blue colors distinguish between diabetic and non-diabetic individuals. The plot does not show a strong linear correlation between BMI and age; however, a higher concentration of diabetic cases is visible among individuals with higher BMI, regardless of age. This observation aligns with the known relationship between higher BMI and increased diabetes risk.  
                
    """)

    

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    st.dataframe(diabetes_df.head(), use_container_width=True, hide_index=True)

    st.markdown("""

    Since the distribution of Iris species in our dataset is **balanced** and there are **0 null values** as well in our dataset. We will be proceeding already with creating the **Embeddings** for the *species* column and **Train-Test split** for training our machine learning model.
         
    """)

    encoder = LabelEncoder()

# 2. Encode the 'Outcome' column and store it in a new column 'Outcome_encoded'
    diabetes_df['Outcome_encoded'] = encoder.fit_transform(diabetes_df['Outcome'])

# 3. Now you can remap the values in the 'Outcome_encoded' column
    diabetes_df['Outcome_encoded'] = diabetes_df['Outcome_encoded'].map({0: 1, 1: 0})

    st.dataframe(diabetes_df.head(), use_container_width=True, hide_index=True)

    st.markdown("""

    Now we converted the values of **species** column to numerical values using `LabelEncoder`. The **species_encoded** column can now be used as a label for training our supervised model.
         
    """)

    # Mapping of the Iris species and their encoded equivalent

    unique_Outcome = diabetes_df['Outcome'].unique()
    unique_Outcome_encoded = diabetes_df['Outcome_encoded'].unique()

    # Create a new DataFrame
    Outcome_mapping_df = pd.DataFrame({'Outcome': unique_Outcome, 'Outcome Encoded': unique_Outcome_encoded})

    # Display the DataFrame
    st.dataframe(Outcome_mapping_df, use_container_width=True, hide_index=True)

    st.markdown("""

    With the help of **embeddings**, Iris-setosa is now represented by a numerical value of **0**, Iris-versicolor represented by **1**, and Iris-virginica represented by **2**.
         
    """)

    st.subheader("Train-Test Split")

    # Select features and target variable
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = diabetes_df[features]
    y = diabetes_df['Outcome_encoded']

    st.code("""

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = diabetes_df[features]
    y = diabetes_df['Outcome_encoded']

            
    """)

    st.markdown("""

    Now we select the features and labels for training our model.  
    The potential `features` that we can use are **sepal_length**, **sepal_width**, **petal_length**, and **petal_width**.  
    As for the `label` we can use the **species_encoded** column derived from the *species* column.
         
    """)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.code("""

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
    """)

    st.subheader("X_train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)

    st.subheader("X_test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.subheader("y_train")
    st.dataframe(y_train, use_container_width=True, hide_index=True)

    st.subheader("y_test")
    st.dataframe(y_test, use_container_width=True, hide_index=True)

    st.markdown("After splitting our dataset into `training` and `test` set. We can now proceed with **training our supervised models**.")

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    st.subheader("Decision Tree Classifier")
    st.markdown("""

    **Decision Tree Classifier** from Scikit-learn library is a machine learning algorithm that is used primarily for *classification* tasks. Its goal is to *categorize* data points into specific classes. This is made by breaking down data into smaller and smaller subsets based on questions which then creates a `"Tree"` structure wherein each **node** in the tree represents a question or decision point based on the feature in the data. Depending on the answer, the data moves down one **branch** of the tree leading to another node with a new question or decision.  

    This process continues until reaching the **leaf** node wherein a class label is assigned. The algorithm then chooses questions that tends to split the data to make it pure at each level.

    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html         
                
    """)

    # Columns to center the Decision Tree Parts image
    col_dt_fig = st.columns((2, 4, 2), gap='medium')

    with col_dt_fig[0]:
        st.write(' ')

    with col_dt_fig[1]:
        decision_tree_parts_image = Image.open('assets/figures/decision_tree_parts.png')
        st.image(decision_tree_parts_image, caption='Decision Tree Parts')

    with col_dt_fig[2]:
        st.write(' ')

    st.subheader("Training the Decision Tree Classifier")

    st.code("""

    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)     
            
    """)

    st.subheader("Model Evaluation")

    st.code("""

    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
            
    """)

    st.write("Accuracy: 70.13%")

    st.markdown("""

    Upon training our Decision Tree classifier model, our model managed to obtain 100% accuracy after the training indicating that it was able to learn and recognize patterns from the dataset.
     
    """)

    st.subheader("Feature Importance")

    st.code("""

    decision_tree_feature_importance = pd.Series(dt_classifier.feature_importances_, index=X_train.columns)

    decision_tree_feature_importance
     
    """)

    dt_feature_importance = {
        'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'Importance': [0.027255, 0.356937, 0.097345, 0.061428, 0.039552, 0.186479, 0.096871, 0.134134]
    }

    dt_feature_importance_df = pd.DataFrame(dt_feature_importance)

    st.dataframe(dt_feature_importance_df, use_container_width=True, hide_index=True)

    feature_importance_plot(dt_feature_importance_df, 500, 500, 1)

    st.markdown("""

    Upon running `.feature_importances` in the `Decision Tree Classifier Model` to check how each Iris species' features influence the training of our model, it is clear that **petal_length** holds the most influence in our model's decisions having **0.89** or **89%** importance. This is followed by **petal_width** which is far behind of petal_length having **0.087** or **8.7%** importance.

    """)

    dt_tree_image = Image.open('assets/model_results/DTTree.png')
    st.image(dt_tree_image, caption='Decision Tree Classifier - Tree Plot')

    st.markdown("""

    This **Tree Plot** visualizes how our **Decision Tree** classifier model makes its predictions based on what was learned from the Iris species' features during the training.
                
    """)
        

    # Logistic Regression

    st.subheader("Logistic Regression")

    st.markdown("""

    **Random Forest Regressor** is a machine learning algorithm that is used to predict continuous values by *combining multiple decision trees* which is called `"Forest"` wherein each tree is trained independently on different random subset of data and features.

    This process begins with data **splitting** wherein the algorithm selects various random subsets of both the data points and the features to create diverse decision trees.  

    Each tree is then trained separately to make predictions based on its unique subset. When it's time to make a final prediction each tree in the forest gives its own result and the Random Forest algorithm averages these predictions.

    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
         
    """)

    # Columns to center the Random Forest Regressor figure image
    col_rfr_fig = st.columns((2, 4, 2), gap='medium')

    with col_rfr_fig[0]:
        st.write(' ')

    with col_rfr_fig[1]:
        decision_tree_parts_image = Image.open('assets/figures/Random-Forest-Figure.png')
        st.image(decision_tree_parts_image, caption='Random Forest Figure')

    with col_rfr_fig[2]:
        st.write(' ')

    st.subheader("Training the Logistic Regression model")

    st.code("""

    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)     
            
    """)

    st.subheader("Model Evaluation")

    st.code("""

    # Evaluate the model
    train_accuracy = log_reg.score(X_train, y_train) #train daTa
    test_accuracy = log_reg.score(X_test, y_test) #test daTa

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    
    """)

    st.write("""

    **Train Accuracy:** 78.40%\n
    **Test Accuracy:** 73.59%      
             
    """)

    st.subheader("Feature Importance")

    st.code("""

    logistic_regression_feature_importance = pd.Series(log_reg.coef_[0], index=X_train.columns)

    logistic_regression_feature_importance
    
    """)

    logistic_regression_feature_importance = {
        'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'Importance': [0.027255, 0.356937, 0.097345, 0.061428, 0.039552, 0.186479, 0.096871, 0.134134]
    }

    logistic_regression_feature_importance_df = pd.DataFrame(logistic_regression_feature_importance)

    st.dataframe(logistic_regression_feature_importance_df, use_container_width=True, hide_index=True)

    feature_importance_plot(logistic_regression_feature_importance_df, 500, 500, 2)

    st.markdown("""

    Upon running `.feature_importances` in the `Random Forest Regressor Model` to check how each Iris species' features influence the training of our model, it is clear that **petal_length** holds the most influence in our model's decisions having **0.58** or **58%** importance. This is followed by **petal_width** which is far behind of petal_length having **0.39** or **39%** importance.

    """)

    st.subheader("Number of Trees")
    st.code("""

    print(f"Number of trees made: {len(rfr_classifier.estimators_)}")
     
    """)

    st.markdown("**Number of trees made:** 100")

    st.subheader("Plotting the Forest")

    forest_image = Image.open('assets/model_results/RFRForest.png')
    st.image(forest_image, caption='Random Forest Regressor - Forest Plot')

    st.markdown("This graph shows **all of the decision trees** made by our **Random Forest Regressor** model which then forms a **Forest**.")

    st.subheader("Forest - Single Tree")

    forest_single_tree_image = Image.open('assets/model_results/RFRTreeOne.png')
    st.image(forest_single_tree_image, caption='Random Forest Regressor - Single Tree')

    st.markdown("This **Tree Plot** shows a single tree from our Random Forest Regressor model.")


# Prediction Page
elif  st.session_state.page_selection == "prediction":
    st.header("Prediction")

    col_pred = st.columns((1.5, 3, 3), gap='medium')

    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False

    with col_pred[0]:
        with st.expander('Options', expanded=True):
            show_dataset = st.checkbox('Show Dataset')
            show_classes = st.checkbox('Show All Classes')
            show_Diabetes = st.checkbox('Show Diabetes')
            show_No_Diabetes = st.checkbox('Show No Diabetes')
            

            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:

                st.session_state.clear = True

    with col_pred[1]:
        st.markdown("#### Decision Tree Classifier")
        
        # Input boxes for the features
        dt_Pregnancies = st.number_input('Pregnancies', min_value=0.0, max_value=10.0, step=0.1, key='dt_Pregnancies', value=0.0 if st.session_state.clear else st.session_state.get('dt_Pregnancies', 0.0))
        dt_Glucose = st.number_input('Glucose', min_value=0.0, max_value=10.0, step=0.1, key='dt_Glucose', value=0.0 if st.session_state.clear else st.session_state.get('dt_Glucose', 0.0))
        dt_BloodPressure = st.number_input('BloodPressure', min_value=0.0, max_value=10.0, step=0.1, key='dt_BloodPressure', value=0.0 if st.session_state.clear else st.session_state.get('dt_BloodPressure', 0.0))
        dt_SkinThickness = st.number_input('SkinThickness', min_value=0.0, max_value=10.0, step=0.1, key='dt_SkinThickness', value=0.0 if st.session_state.clear else st.session_state.get('dt_SkinThickness', 0.0))
        dt_Insulin = st.number_input('Insulin', min_value=0.0, max_value=10.0, step=0.1, key='dt_Insulin', value=0.0 if st.session_state.clear else st.session_state.get('dt_Insulin', 0.0))
        dt_BMI = st.number_input('BMI', min_value=0.0, max_value=10.0, step=0.1, key='dt_BMI', value=0.0 if st.session_state.clear else st.session_state.get('dt_BMI', 0.0))
        dt_DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=10.0, step=0.1, key='dt_DiabetesPedigreeFunction', value=0.0 if st.session_state.clear else st.session_state.get('dt_DiabetesPedigreeFunction', 0.0))
        dt_Age = st.number_input('Age', min_value=0.0, max_value=10.0, step=0.1, key='dt_Age', value=0.0 if st.session_state.clear else st.session_state.get('dt_Age', 0.0))

        classes_list = ['Diabetes', 'No Diabetes']
        
        # Button to detect the Iris species
        if st.button('Detect', key='dt_detect'):
            # Prepare the input data for prediction
            dt_input_data = [[dt_Pregnancies, dt_Glucose, dt_BloodPressure, dt_SkinThickness, dt_Insulin, dt_BMI, dt_DiabetesPedigreeFunction, dt_Age]]
            
            # Predict the Iris species
            dt_prediction = dt_classifier.predict(dt_input_data)
            
            # Display the prediction result
            st.markdown(f'The predicted outcome is: `{classes_list[dt_prediction[0]]}`')

    with col_pred[2]:
        st.markdown("#### Logistic Regression")

         # Input boxes for the features
        log_Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=100, step=1, key='log_Pregnancies', value=0 if st.session_state.clear else st.session_state.get('log_Pregnancies', 0.0))
        log_Glucose = st.number_input('Glucose', min_value=0.0, max_value=10.0, step=0.1, key='log_Glucose', value=0.0 if st.session_state.clear else st.session_state.get('log_Glucose', 0.0))
        log_BloodPressure = st.number_input('BloodPressure', min_value=0.0, max_value=10.0, step=0.1, key='log_BloodPressure', value=0.0 if st.session_state.clear else st.session_state.get('log_BloodPressure', 0.0))
        log_SkinThickness = st.number_input('SkinThickness', min_value=0.0, max_value=10.0, step=0.1, key='log_SkinThickness', value=0.0 if st.session_state.clear else st.session_state.get('log_SkinThickness', 0.0))
        log_Insulin = st.number_input('Insulin', min_value=0.0, max_value=10.0, step=0.1, key='log_Insulin', value=0.0 if st.session_state.clear else st.session_state.get('log_Insulin', 0.0))
        log_BMI = st.number_input('BMI', min_value=0.0, max_value=10.0, step=0.1, key='log_BMI', value=0.0 if st.session_state.clear else st.session_state.get('log_BMI', 0.0))
        log_DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=10.0, step=0.1, key='log_DiabetesPedigreeFunction', value=0.0 if st.session_state.clear else st.session_state.get('log_DiabetesPedigreeFunction', 0.0))
        log_Age = st.number_input('Age', min_value=0.0, max_value=10.0, step=0.1, key='log_Age', value=0.0 if st.session_state.clear else st.session_state.get('log_Age', 0.0))

        classes_list = ['Diabetes', 'No Diabetes']
        
        # Button to detect the Iris species
        if st.button('Detect', key='log_detect'):
            # Prepare the input data for prediction
            log_input_data = [[log_Pregnancies, log_Glucose, log_BloodPressure, log_SkinThickness, log_Insulin, log_BMI, log_DiabetesPedigreeFunction, log_Age]]
            
            # Predict the Iris species
            log_prediction = log_reg.predict(log_input_data)
            
            # Display the prediction result
            st.markdown(f'The predicted outcome is: `{classes_list[log_prediction[0]]}`')

    # Create 3 Data Frames containing  5 rows for each species
    Diabetes_samples = diabetes_df[diabetes_df["Outcome"] == "Diabetes"].head(5)
    No_Diabetes_samples = diabetes_df[diabetes_df["Outcome"] == "No Diabetes"].head(5)
    

    if show_dataset:
        # Display the dataset
        st.subheader("Dataset")
        st.dataframe(diabetes_df, use_container_width=True, hide_index=True)

    if show_classes:
        # Diabetes Samples
        st.subheader("Diabetes Samples")
        st.dataframe(Diabetes_samples, use_container_width=True, hide_index=True)

        # No Diabetes Samples
        st.subheader("No Diabetes Samples")
        st.dataframe(No_Diabetes_samples, use_container_width=True, hide_index=True)

    if show_Diabetes:
        # Display the Diabetes samples
        st.subheader("Diabetes Samples")
        st.dataframe(Diabetes_samples, use_container_width=True, hide_index=True)

    if show_No_Diabetes:
        # Display the No Diabetes samples
        st.subheader("No Diabetes Samples")
        st.dataframe(No_Diabetes_samples, use_container_width=True, hide_index=True)

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""
                
    Through exploratory data analysis and training of two classification models (`Decision Tree Classifier` and `Random Forest Regressor`) on the **Iris Flower dataset**, the key insights and observations are:

    #### 1. 📊 **Dataset Characteristics**:
    - The dataset shows moderate variation across the **sepal and petal** features. `petal_length` and `petal_width` has higher variability than the sepal features further suggesting that these features are more likely to distinguish between the three Iris flower species.
    - All of the three Iris species have a **balanced class distribution** which further eliminates the need to rebalance the dataset.

    #### 2. 📝 **Feature Distributions and Separability**:
    - **Pairwise Scatter Plot** analysis indicates that `Iris Setosa` forms a distinct cluster based on petal features which makes it easily distinguishable from `Iris Versicolor` and `Iris Virginica`.
    - **Petal Length** emerged as the most discriminative feature especially for distinguishing `Iris Setosa` from other Iris species.

    #### 3. 📈 **Model Performance (Decision Tree Classifier)**:

    - The `Decision Tree Classifier` achieved 100% accuracy on the training data which suggests that using a relatively simple and structured dataset resulted in a strong performance for this model. However, this could also imply potential **overfitting** due to the model's high sensitivity to the specific training samples.
    - In terms of **feature importance** results from the *Decision Tree Model*, `petal_length` was the dominant predictor having **89%** importance value which is then followed by `petal_width` with **8.7%**.

    #### 4. 📈 **Model Performance (Random Forest Regressor)**:
    - The **Random Forest Regressor** achieved an accuracy of 98.58% on training and 99.82% on testing which is slightly lower compared to the performance of the *Decision Tree Classifier Model*
    - **Feature importance** analysis also highlighted `petal_length` as the primary predictor having **58%** importance value followed by `petal_width` with **39%**.

    ##### **Summing up:**  
    Throughout this data science activity, it is evident that the Iris dataset is a good dataset to use for classification despite of its simplicity. Due to its balanced distribution of 3 Iris flower species and having 0 null values, further data cleansing techniques were not used. 2 of the classifier models trained were able to leverage the features that can be found in the dataset which resulted to a high accuracy in terms of the two models' predictions. Despite of the slight overlap between Iris Versicolor and Iris Virginica, the two models trained were able to achieve high accuracy and was able to learn patterns from the dataset.         
                
    """)