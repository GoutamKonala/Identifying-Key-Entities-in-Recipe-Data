# Identifying Key Entities in Recipe Data

This project focuses on training a Named Entity Recognition (NER) model using Conditional Random Fields (CRF) to extract key entities from recipe data. The model classifies words into predefined categories: **ingredients**, **quantities**, and **units**.

## Business Objective
The goal is to create a structured database of recipes and ingredients that can power advanced features in recipe management systems, dietary tracking apps, or e-commerce platforms.

## Dataset
The dataset is provided in `ingredient_and_quantity.json` and consists of structured recipe ingredient lists with NER labels.
- `input`: Raw ingredient list from a recipe.
- `pos`: Corresponding Part-of-Speech (POS) tags or NER labels (quantity, ingredient, unit).

## Prerequisites
The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- sklearn-crfsuite
- scikit-learn
- spacy
- joblib

You may need to install `sklearn-crfsuite` and download the spacy model:
```bash
pip install sklearn-crfsuite spacy pandas numpy matplotlib seaborn scikit-learn joblib
python -m spacy download en_core_web_sm
```

## Project Workflow

1.  **Data Ingestion and Preparation**:
    - Load data from JSON.
    - Tokenize `input` and `pos` fields.
    - Validate data integrity (matching lengths) and clean invalid entries.
    - Split data into training (70%) and validation (30%) sets.

2.  **Exploratory Data Analysis (EDA)**:
    - Analyze distribution of tokens (Ingredients, Units, Quantities).
    - Visualize top frequent items.

3.  **Feature Engineering**:
    - Extract features for each token, including:
        - Core features: word lower, lemma, POS tags, shapes, digit checks.
        - Domain-specific features: `is_quantity`, `is_unit` (using keyword lists and regex).
        - Contextual features: Previous and next token attributes.
    - Compute class weights based on inverse frequency to handle class imbalance.

4.  **Model Building**:
    - Initialize and train a CRF model (`sklearn_crfsuite.CRF`) using the weighted features.
    - Hyperparameters: `lbfgs` algorithm, L1/L2 regularization.

5.  **Evaluation**:
    - Evaluate model performance using Flat F1-score, Precision, Recall, and Confusion Matrix.
    - Perform Error Analysis on the validation set to understand misclassifications.

## Files
- `Identifying_Key_Entities_in_Recipe_Data_Lakshmi_Goutam_Reddy_Konala.ipynb`: The main Jupyter Notebook containing the code and analysis.
- `ingredient_and_quantity.json`: The dataset used for training and evaluation.

## Results
The model achieves high accuracy in identifying recipe entities. Detailed metrics and error analysis are provided within the notebook.
