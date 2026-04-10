# SkillCraft Internship - Task 01: House Price Prediction

## 📌 Project Overview
This project is part of the SkillCraft Internship program. The goal is to build a **Linear Regression model** that predicts house prices using the provided training dataset and generate predictions for the test dataset in the required submission format.

---

## 📂 Dataset
- **train.csv** → Contains features and target (`SalePrice`) for training the model.
- **test.csv** → Contains features without target; predictions must be generated.
- **submission.csv** → Final output file with columns:
  - `Id`
  - `SalePrice`

---

## 🔑 Features Used
The following features were selected based on their relevance to house pricing:

- `OverallQual` – Overall quality rating  
- `GrLivArea` – Living area above ground (sq ft)  
- `GarageCars` – Garage capacity (number of cars)  
- `GarageArea` – Garage size (sq ft)  
- `TotalBsmtSF` – Basement area (sq ft)  
- `YearBuilt` – Year the house was built  
- `YearRemodAdd` – Year of remodel  
- `FullBath` – Number of full bathrooms  
- `HalfBath` – Number of half bathrooms  
- `TotRmsAbvGrd` – Total rooms above ground  
- `LotArea` – Lot size (sq ft)  

---

## ⚙️ Steps
1. Load training and test datasets using Pandas.  
2. Select relevant numeric features.  
3. Handle missing values by filling with `0`.  
4. Train a **Linear Regression** model on the training data.  
5. Evaluate the model using Mean Squared Error (MSE) and R² Score.  
6. Generate predictions for the test dataset.  
7. Save predictions in `submission.csv` with required format (`Id, SalePrice`).  

---

## 📊 Visualizations
- **Actual vs Predicted Scatter Plot**: Shows prediction accuracy.  

---

## 🚀 How to Run
1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
2. Place `house_prices_train.csv` and `house_prices_test.csv` inside the `Task1` folder.  
3. Run the notebook/script:
   ```bash
   python ML_Task1.py
   ```
4. The output file `submission.csv` will be generated.

---

## 📈 Results
- Model trained successfully on selected features.  
- Predictions generated for the test dataset.  
- Submission file created in the required format.  
