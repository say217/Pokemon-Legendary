from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def train_random_forest(X, y, n_estimators=200, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state
    )
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, scaler, metrics

def plot_random_forest_tree(model, tree_index=0, save_path=None):
    tree = model.estimators_[tree_index]
    plt.figure(figsize=(30, 20))
    plot_tree(tree, filled=True, rounded=True, fontsize=10)
    plt.title(f'Random Forest Tree #{tree_index + 1}', fontsize=20)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
