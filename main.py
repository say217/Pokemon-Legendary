
from data_preprocessing import load_data, clean_and_select_columns, encode_abilities
from feature_engineering import encode_classification, encode_type1
from visualization import scatter_sp_attack_defense, histogram_against_columns, scatter_3d
from modeling import train_random_forest, plot_random_forest_tree

# Load & preprocess
selected_columns = [
    "abilities","against_bug","against_dark","against_dragon","against_electric",
    "against_fairy","against_fight","against_fire","against_flying","against_ghost",
    "against_grass","against_ground","against_ice","against_normal","against_poison",
    "against_psychic","against_rock","against_steel","against_water","attack",
    "classfication","defense","experience_growth","hp","pokedex_number","sp_attack",
    "sp_defense","speed","type1","generation","is_legendary"
]

df = load_data("data/pokemon.csv")
df = clean_and_select_columns(df, selected_columns)
df = encode_abilities(df)
df = encode_classification(df)
df = encode_type1(df)

# Visualizations
scatter_sp_attack_defense(df)
against_cols = [col for col in df.columns if 'against' in col]
histogram_against_columns(df, against_cols)
scatter_3d(df, 'speed', 'generation', 'hp', color_col='generation', size_col='hp', title="Speed vs Generation vs HP")

# Modeling
X = df.drop(columns=["is_legendary"])
y = df["is_legendary"]
model, scaler, metrics = train_random_forest(X, y)
print(metrics)
plot_random_forest_tree(model, tree_index=0, save_path='random_forest_tree_1.png')




