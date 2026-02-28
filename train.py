import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('Obesity.csv')

cols_to_round = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in cols_to_round:
    df[col] = df[col].round().astype(int)

# pegando Variavel Target
X = df.drop('Obesity', axis=1)
y = df['Obesity']

num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
cat_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# Construção da Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Treino do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Verificando Assertividade
acc = accuracy_score(y_test, model_pipeline.predict(X_test))
print(f"Modelo treinado com sucesso. Accuracy: {acc:.2%}")

# Salvando
joblib.dump(model_pipeline, 'model_obesity.pkl')