# Iris Classification Dashboard using Streamlit

A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to classify Iris species from the Iris dataset (Setosa, Versicolor, and Virginica) using **Decision Tree Classifier** and **Random Forest Regressor**.

![Main Page Screenshot](screenshots/IrisClassificationDashboard.webp)

### 🔗 Links:

- 🌐 [Streamlit Link](https://zeraphim-iris-classification-dashboard.streamlit.app/)
- 📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)

### 📊 Dataset:

- [Iris Flower Dataset (Kaggle)](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

### 📖 Pages:

1. `Dataset` - Brief description of the Iris Flower dataset used in this dashboard.
2. `EDA` - Exploratory Data Analysis of the Iris Flower dataset. Highlighting the distribution of Iris species and the relationship between the features. Includes graphs such as Pie Chart, Scatter Plots, and Pairwise Scatter Plot Matrix.
3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets.
4. `Machine Learning` - Training two supervised classification models: Decision Tree Classifier and Random Forest Regressor. Includes model evaluation, feature importance, and tree plot.
5. `Prediction` - Prediction page where users can input values to predict the Iris species using the trained models.
6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

### 💡 Findings / Insights

Through exploratory data analysis and training of two classification models (`Decision Tree Classifier` and `Random Forest Regressor`) on the **Iris Flower dataset**, the key insights and observations are:

#### 1. 📊 **Dataset Characteristics**:

- The dataset shows moderate variation across the **sepal and petal** features. `petal_length` and `petal_width` has higher variability than the sepal features further suggesting that these features are more likely to distinguish between the three Iris flower species.
- All of the three Iris species have a **balanced class distribution** which further eliminates the need to rebalance the dataset.

#### 2. 📝 **Feature Distributions and Separability**:

- **Pairwise Scatter Plot** analysis indicates that `Iris Setosa` forms a distinct cluster based on petal features which makes it easily distinguishable from `Iris Versicolor` and `Iris Virginica`.
- **Petal Length** emerged as the most discriminative feature especially for distinguishing `Iris Setosa` from other Iris species.

#### 3. 📈 **Model Performance (Decision Tree Classifier)**:

- The `Decision Tree Classifier` achieved 100% accuracy on the training data which suggests that using a relatively simple and structured dataset resulted in a strong performance for this model. However, this could also imply potential **overfitting** due to the model's high sensitivity to the specific training samples.
- In terms of **feature importance** results from the _Decision Tree Model_, `petal_length` was the dominant predictor having **89%** importance value which is then followed by `petal_width` with **8.7%**.

#### 4. 📈 **Model Performance (Random Forest Regressor)**:

- The **Random Forest Regressor** achieved an accuracy of 98.58% on training and 99.82% on testing which is slightly lower compared to the performance of the _Decision Tree Classifier Model_
- **Feature importance** analysis also highlighted `petal_length` as the primary predictor having **58%** importance value followed by `petal_width` with **39%**.

##### **Summing up:**

Throughout this data science activity, it is evident that the Iris dataset is a good dataset to use for classification despite of its simplicity. Due to its balanced distribution of 3 Iris flower species and having 0 null values, further data cleansing techniques were not used. 2 of the classifier models trained were able to leverage the features that can be found in the dataset which resulted to a high accuracy in terms of the two models' predictions. Despite of the slight overlap between Iris Versicolor and Iris Virginica, the two models trained were able to achieve high accuracy and was able to learn patterns from the dataset.
