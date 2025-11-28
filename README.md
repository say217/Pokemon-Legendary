<img width="1600" height="1120" alt="image" src="https://github.com/user-attachments/assets/d346a65b-f9db-4152-b051-ec933920890c" />


# Pokémon Legendary Classification Project

This project focuses on building a **classification model** to predict whether a Pokémon is **legendary or non-legendary** based on its attributes. The dataset contains 801 Pokémon with detailed stats, types, abilities, and other battle-related characteristics.

---

## 📌 Project Overview

This machine learning project aims to:

* Explore and clean the Pokémon dataset.
* Perform feature engineering (including multi-label ability encoding).
* Visualize statistical patterns using matplotlib and seaborn.
* Build and evaluate ML models to classify Pokémon as **legendary** or **non-legendary**.

---

## 📂 Dataset Details

The dataset contains **41 columns**, including:

* **Basic Stats:** attack, defense, hp, speed
* **Types & Abilities:** type1, type2, abilities
* **Damage Multipliers:** against_fire, against_water, etc.
* **Metadata:** generation, classfication, experience_growth
* **Target Column:** `is_legendary`

---

## 🔧 Key Steps in the Notebook

### 1. Data Loading & Inspection

* Loaded the dataset using pandas.
* Checked missing values and general information.

### 2. Data Cleaning

* Selected relevant features.
* Removed rows with unnecessary data.
* Extracted ability lists using `ast.literal_eval`.

### 3. Feature Engineering

* Applied **MultiLabelBinarizer** to one-hot encode Pokémon abilities.
* Merged encoded features with the main dataframe.

### Data Visualization

Used matplotlib & seaborn to analyze:

* Type distribution
* Stat comparison between legendary and non-legendary Pokémon
* Correlation heatmap
* Attack/Defense/Speed distributions

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/21e3ed33-3f62-4896-a67e-75490537d7bb" alt="Image 1" width="150px" height="100px" style="object-fit: cover; border-radius: 5px;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/83ea7c47-14ef-416f-8a83-3cba23022f85" alt="Image 2" width="150px" height="100px" style="object-fit: cover; border-radius: 5px;">
    </td>
  </tr>
</table>




### 5. Model Building

You may include the models you used:

* Logistic Regression
* Random Forest
* XGBoost
* SVM
  (Evaluations such as accuracy, confusion matrix, ROC curve)

---

## 📊 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 How to Run the Project

1. Clone the repository:

```
git clone https://github.com/say217/Pokemon-Legendary.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open the Jupyter Notebook:

```
jupyter notebook Pokemon classification.ipynb
```

---

## 📈 Results & Insights

You can describe your findings here, such as:

* Which stats influence legendary status
* Best-performing model
* Accuracy achieved

---

## 📜 Future Improvements

* Add deep learning model (TensorFlow/PyTorch)
* Improve feature selection
* Visualize more advanced Pokémon patterns

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork and raise pull requests.

---

## ⭐ If you like this project

Give a star ⭐ on GitHub to support the work!

---

## 📬 Contact

**Author:** Sayak Samanta
GitHub: [https://github.com/say217](https://github.com/say217)
LinkedIn: Add your profile link here.








<img width="1000" height="500" alt="newplot (7)" src="https://github.com/user-attachments/assets/83ea7c47-14ef-416f-8a83-3cba23022f85" />
<img width="1000" height="500" alt="newplot (8)" src="https://github.com/user-attachments/assets/21e3ed33-3f62-4896-a67e-75490537d7bb" />
<img width="2345" height="1590" alt="random_forest_tree_1" src="https://github.com/user-attachments/assets/467dc1da-ac7b-42b9-a86b-f983a0bf1431" />
<img width="837" height="357" alt="Screenshot 2025-11-27 014238" src="https://github.com/user-attachments/assets/1c46286d-dc55-4d1e-af2c-75099c1f72c5" />
<img width="833" height="360" alt="Screenshot 2025-11-27 014201" src="https://github.com/user-attachments/assets/ad03c313-2003-4323-a666-b8c4b78c9502" />
<img width="532" height="482" alt="Screenshot 2025-11-27 014136" src="https://github.com/user-attachments/assets/eb43064d-29d8-4ef1-b7da-9c3a41bf9132" />

