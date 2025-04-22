# Vegetable Classification using Machine Learning

This project utilizes machine learning to classify vegetables into three categories — **Leafy**, **Root**, or **Fruit** — based on their nutritional values, including Vitamin A, Vitamin C, and Fiber content. The objective is to develop a model capable of automatically classifying vegetables by analyzing their nutritional composition.

## Problem Statement

The goal of this project is to build a machine learning model that can classify vegetables based on key nutritional features. This tool can benefit various applications such as:

- **Nutritional Planning**: Assisting individuals in planning balanced diets.
- **Diet Tracking Apps**: Helping users track their vegetable intake by classifying them automatically.
- **Smart Farming Tools**: Aiding in the classification of vegetables in farming environments based on nutritional data.

## Machine Learning Approach

For this project, we use the **Random Forest Classifier**, a robust ensemble learning method that leverages multiple decision trees to improve classification accuracy. Random Forest ensures better performance by averaging the predictions from a variety of decision trees, reducing overfitting and enhancing model stability.

## Tools & Libraries

This project was developed using the following tools and libraries:

- **Python**: The primary programming language for this project.
- **pandas**: Used for data manipulation and preprocessing.
- **scikit-learn**: Provides tools for model building, training, and evaluation.
- **seaborn & matplotlib**: Used for data visualization and generating plots such as confusion matrices.

## Dataset

The dataset used for this project contains the following columns:

- **vitamin_a**: The amount of Vitamin A in the vegetable (numeric value).
- **vitamin_c**: The amount of Vitamin C in the vegetable (numeric value).
- **fiber**: The amount of fiber content in the vegetable (numeric value).
- **type**: The target label that indicates the category of the vegetable (Leafy, Root, Fruit).

**File**: `vegetables.csv`

## Steps Performed

1. **Load and Explore the Dataset**:
   - The dataset was loaded and initial exploration was performed to understand its structure and key features.

2. **Preprocess Data**:
   - Data preprocessing steps were carried out, including label encoding for the target variable and splitting the dataset into training and testing sets.

3. **Train Random Forest Classifier**:
   - The Random Forest Classifier was trained on the dataset to learn the relationships between the nutritional features and vegetable categories.

4. **Evaluate the Model**:
   - The model's performance was evaluated using several metrics, including:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**
   
5. **Visualize the Confusion Matrix**:
   - A heatmap of the confusion matrix was generated to visually assess the model’s classification performance.

## Results

- **High Accuracy**: The model demonstrated high accuracy in classifying vegetables into their respective categories.
- **Balanced Performance**: The classification report showed that the model performs well across all categories (Leafy, Root, and Fruit).
- **Confusion Matrix Visualization**: The confusion matrix heatmap provided a clear visualization of model performance across different classes, helping to identify any misclassifications.

## Example Output

- **Clean Classification Report**: A detailed breakdown of precision, recall, and F1-score for each vegetable category.
- **Heatmap of Confusion Matrix**: A visual representation of the model’s classification results.
