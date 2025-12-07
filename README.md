 
# Pokémon Legendary Classification Project

This project focuses on building a **classification model** to predict whether a Pokémon is **legendary or non-legendary** based on its attributes. The dataset contains 801 Pokémon with detailed stats, types, abilities, and other battle-related characteristics.
## Technologies Used

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-F3766E?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)

<img width="1600" height="1120" alt="image" src="https://github.com/user-attachments/assets/d346a65b-f9db-4152-b051-ec933920890c" />

---

## Project Overview

This machine learning project aims to:

* Explore and clean the Pokémon dataset.
* Perform feature engineering (including multi-label ability encoding).
* Visualize statistical patterns using matplotlib and seaborn.
* Build and evaluate ML models to classify Pokémon as **legendary** or **non-legendary**.

---

## Dataset Details
This Pokémon dataset contains 801 entries and 41 columns, offering a rich combination of battle stats, typing information, resistance multipliers, biological traits, and metadata. The abilities column lists each Pokémon’s unique powers (e.g., Overgrow, Blaze, Beast Boost) and is stored as an object containing ability lists. A major portion of the dataset consists of damage-multiplier attributes such as against_fire, against_water, against_fairy, and others—each represented as a float showing how much damage the Pokémon receives from different attack types. These multipliers act as the Pokémon’s strengths and weaknesses, with values like 0.5 (resistant), 2.0 (weak) or 0.0 (immune) forming a key part of battle performance.

The dataset also includes all essential battle statistics—attack, defense, hp, sp_attack, sp_defense, and speed—all stored as integers. Together with the Pokémon’s base_total, these features describe its overall combat capability and are vital for evaluating performance.

| abilities                      | against_bug | against_dark | against_dragon | against_electric | against_fairy | against_fight | against_fire | against_flying | against_ghost | percentage_male | pokedex_number | sp_attack | sp_defense | speed | type1 | type2  | weight_kg | generation | is_legendary |
|--------------------------------|-------------|--------------|----------------|------------------|---------------|----------------|--------------|----------------|----------------|------------------|----------------|-----------|------------|--------|--------|--------|-----------|------------|--------------|
| ['Overgrow', 'Chlorophyll']    | 1.00        | 1.0          | 1.0            | 0.5              | 0.5           | 0.5            | 2.0          | 2.0            | 1.0            | 88.1             | 1              | 65        | 65         | 45     | grass  | poison | 6.9       | 1          | 0            |
| ['Overgrow', 'Chlorophyll']    | 1.00        | 1.0          | 1.0            | 0.5              | 0.5           | 0.5            | 2.0          | 2.0            | 1.0            | 88.1             | 2              | 80        | 80         | 60     | grass  | poison | 13.0      | 1          | 0            |
| ['Overgrow', 'Chlorophyll']    | 1.00        | 1.0          | 1.0            | 0.5              | 0.5           | 0.5            | 2.0          | 2.0            | 1.0            | 88.1             | 3              | 122       | 120        | 80     | grass  | poison | 100.0     | 1          | 0            |
| ['Blaze', 'Solar Power']       | 0.50        | 1.0          | 1.0            | 1.0              | 0.5           | 1.0            | 0.5          | 1.0            | 1.0            | 88.1             | 4              | 60        | 50         | 65     | fire   | NaN    | 8.5       | 1          | 0            |
| ['Blaze', 'Solar Power']       | 0.50        | 1.0          | 1.0            | 1.0              | 0.5           | 1.0            | 0.5          | 1.0            | 1.0            | 88.1             | 5              | 80        | 65         | 80     | fire   | NaN    | 19.0      | 1          | 0            |


<img width="837" height="357" alt="Screenshot 2025-11-27 014238" src="https://github.com/user-attachments/assets/1c46286d-dc55-4d1e-af2c-75099c1f72c5" />
<img width="833" height="360" alt="Screenshot 2025-11-27 014201" src="https://github.com/user-attachments/assets/ad03c313-2003-4323-a666-b8c4b78c9502" />
<img width="532" height="482" alt="Screenshot 2025-11-27 014136" src="https://github.com/user-attachments/assets/eb43064d-29d8-4ef1-b7da-9c3a41bf9132" />

---

## Data Loading & Inspection

The dataset was loaded into a **pandas DataFrame** for analysis. After loading:

- `.info()` was used to inspect column types and overall structure.
- `.isna().sum()` was used to detect missing values.
- Initial exploration showed that some columns contain missing values, such as:
  - `height_m`
  - `weight_kg`
  - `percentage_male`
- The `abilities` column was found to be stored as **string representations of Python lists**, requiring conversion before use.

This initial inspection ensures a solid understanding of the dataset's structure before applying preprocessing steps.

---

## Feature Engineering

### **One-Hot Encoding of Abilities**
- The `abilities` column contains multiple abilities per Pokémon (e.g., `['Overgrow', 'Chlorophyll']`).
- These were converted into actual Python lists using `ast.literal_eval`.
- `MultiLabelBinarizer` was applied to transform these lists into a **binary one-hot encoded matrix**, making them suitable for machine learning models.

### **Feature Integration**
- The resulting encoded ability features were merged back into the main DataFrame.
- This enriched the dataset with **machine-readable numeric ability features**, enabling models to learn patterns from Pokémon abilities.

---

### Data Visualization

Used Plotly to analyze:

* Type distribution
* Stat comparison between legendary and non-legendary Pokémon
* Correlation heatmap
* Attack/Defense/Speed distributions

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/21e3ed33-3f62-4896-a67e-75490537d7bb" alt="Image 1" width="550px" style="object-fit: cover; border-radius: 5px;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/83ea7c47-14ef-416f-8a83-3cba23022f85" alt="Image 2" width="550px" style="object-fit: cover; border-radius: 5px;">
    </td>
  </tr>
</table>




### 5. Model Building

**Algorithm Used:** Random Forest

**What is Random Forest?**  
  Random Forest is an **ensemble machine learning algorithm** that builds multiple decision trees and merges their predictions to improve accuracy and reduce overfitting.  
  Each tree is trained on a random subset of the data (bagging).  
  At each split in a tree, a random subset of features is considered (feature randomness).  
  The final prediction is obtained by **majority voting** (for classification) or **averaging** (for regression).  

**Why Random Forest for Pokémon Classification?**  
  Handles high-dimensional data well, such as one-hot encoded abilities.  
  Robust to noise and less likely to overfit compared to a single decision tree.  
  Provides feature importance, which helps in understanding which abilities or attributes most influence the prediction.




<img width="2345" height="1290" alt="random_forest_tree_1" src="https://github.com/user-attachments/assets/467dc1da-ac7b-42b9-a86b-f983a0bf1431" />

---



---

## How to Run the Project
```bash
1. Clone the repository:
git clone https://github.com/say217/Pokemon-Legendary.git
2. Install dependencies
pip install -r requirements.txt
3. Open the Jupyter Notebook:
jupyter notebook Pokemon classification.ipynb

```
---
## Results & Insights

After training a **Random Forest Regressor** on the Pokémon dataset, we evaluated the model using standard regression metrics.  
Although the target variable (`is_legendary`) is binary, the regression approach still provides meaningful insights into predictive performance.

### Model Evaluation Metrics

| Metric | Value |
|--------|--------|
| **MSE**  | 0.03246 |
| **RMSE** | 0.18016 |
| **MAE**  | 0.06115 |
| **R²**   | 0.67315 |

### Interpretation of Results

- **Low MSE and RMSE** indicate that the model's predictions are close to the true values.
- **MAE ≈ 0.06** means the model's average error is very small (on a 0–1 scale), which is excellent for a binary-like target.
- **R² ≈ 0.67** suggests that the model explains **67% of the variance** in whether a Pokémon is legendary or not.



## Contribution

Contributions are welcome! Feel free to fork and raise pull requests. If you like this project Give a star ⭐ on GitHub to support the work!

---

## 📬 Contact

**Author:** Sayak Samanta
GitHub: [https://github.com/say217](https://github.com/say217)




