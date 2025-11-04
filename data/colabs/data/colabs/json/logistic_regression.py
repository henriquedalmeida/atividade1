import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

TARGET_LOG = 'resultado'
# DEFINA AQUI O CAMINHO CORRETO PARA O SEU CSV!
FILE_PATH = r'C:\Users\pipoc\Documents\Data_Felipe\UFAPE\RNN - FM\atividade1-parte1\data\processed\dallas_games_2024-25.csv'

# --- 1. CARREGAR O DATAFRAME ---
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo CSV não encontrado em {FILE_PATH}")
    exit()

if TARGET_LOG not in df.columns:
    print(f"\n❌ ERRO CRÍTICO: A coluna '{TARGET_LOG}' não foi encontrada no CSV.")
    exit()

# --- 2. ANÁLISE DE CORRELAÇÃO (IMPRESSÃO 1) ---
correlation_matrix_log = df.corr(numeric_only=True)
target_correlation_log = correlation_matrix_log[TARGET_LOG].sort_values(ascending=False)

print(f"\n--- Coeficientes de Correlação com a Variável TARGET ({TARGET_LOG}) ---")
print(target_correlation_log.drop(TARGET_LOG, errors='ignore'))


# --- 3. CONFIGURAÇÃO DO MODELO LOGÍSTICO ---
# Features selecionadas após exclusão de leakage (saldo-pontos, pontos)
FEATURES_LOG = [
    'porcentagem-triplos',
    'triplos-convertidos',
    'rebotes-totais',
    'porcentagem-arremessos',
    'assistencias',
    'mando-de-jogo'
]

# X e Y do DataFrame 'df'
X = df[FEATURES_LOG]
y = df[TARGET_LOG]

# Divisão em Treino e Teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TREINAMENTO DO MODELO (IMPRESSÃO 2) ---
model_log = LogisticRegression(solver='liblinear', random_state=42)
model_log.fit(X_train, y_train)

# AVALIAÇÃO E COEFICIENTES
y_pred = model_log.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
intercepto = model_log.intercept_[0]

print(f"\n--- Resultados do Modelo de Regressão Logística ---")
print(f"Acurácia (Teste): {accuracy:.4f}")
print(f"Intercepto (Beta_0): {intercepto:.4f}")

# Coeficientes
coef_log = pd.Series(model_log.coef_[0], index=FEATURES_LOG)
print("\nCoeficientes (Beta_n):")
print(coef_log.sort_values(ascending=False))

# --- 5. MÉTRICAS ADICIONAIS ---
# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"\n--- Matriz de Confusão ---")
print(f"                 Previsto")
print(f"              Derrota  Vitória")
print(f"Real Derrota    {conf_matrix[0,0]:3d}      {conf_matrix[0,1]:3d}")
print(f"     Vitória    {conf_matrix[1,0]:3d}      {conf_matrix[1,1]:3d}")

# Relatório de classificação
class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print(f"\n--- Métricas de Classificação ---")
if '0' in class_report:
    print(f"Classe 0 (Derrota):")
    print(f"  Precision: {class_report['0']['precision']:.2%}")
    print(f"  Recall: {class_report['0']['recall']:.2%}")
    print(f"  F1-Score: {class_report['0']['f1-score']:.2%}")
if '1' in class_report:
    print(f"Classe 1 (Vitória):")
    print(f"  Precision: {class_report['1']['precision']:.2%}")
    print(f"  Recall: {class_report['1']['recall']:.2%}")
    print(f"  F1-Score: {class_report['1']['f1-score']:.2%}")

# --- 6. EQUAÇÃO DO MODELO ---
print(f"\n--- Equacao do Modelo de Regressao Logistica ---")
print(f"p(Vitoria) = 1 / [1 + e^(-(B0 + B1*X1 + B2*X2 + ... + Bn*Xn))]")
print(f"\nOnde:")
print(f"  B0 (Intercepto) = {intercepto:.4f}")
for i, feature in enumerate(FEATURES_LOG):
    print(f"  B{i+1} ({feature}) = {model_log.coef_[0][i]:.4f}")

# Exemplo de equação com valores
equation_parts = [f"{intercepto:.4f}"]
for i, feature in enumerate(FEATURES_LOG):
    coef = model_log.coef_[0][i]
    sign = "+" if coef >= 0 else ""
    equation_parts.append(f"{sign}{coef:.4f}*{feature}")
equation = " ".join(equation_parts)
print(f"\nEquacao completa:")
print(f"z = {equation}")
print(f"p(Vitoria) = 1 / [1 + e^(-z)]")

# --- 7. EXEMPLOS DE PREDIÇÕES COM PROBABILIDADES ---
print(f"\n--- Exemplos de Predições com Probabilidades ---")
print("="*80)

# Calcular probabilidades
y_pred_proba = model_log.predict_proba(X_test)

# Pegar alguns exemplos do conjunto de teste
n_examples = min(10, len(X_test))
example_indices = np.random.choice(len(X_test), size=n_examples, replace=False)

examples_list = []
for i, idx in enumerate(example_indices):
    proba_vitoria = y_pred_proba[idx][1]
    proba_derrota = y_pred_proba[idx][0]
    real_resultado = "Vitoria" if y_test.iloc[idx] == 1 else "Derrota"
    pred_resultado = "Vitoria" if y_pred[idx] == 1 else "Derrota"
    correct = "OK" if real_resultado == pred_resultado else "ERRO"

    print(f"\nExemplo {i+1}: {correct}")
    print(f"  Probabilidade de Vitoria: {proba_vitoria:.1%}")
    print(f"  Probabilidade de Derrota: {proba_derrota:.1%}")
    print(f"  Predicao: {pred_resultado}")
    print(f"  Resultado Real: {real_resultado}")

    # Guardar para JSON
    examples_list.append({
        'example_number': i + 1,
        'probability_win': float(proba_vitoria),
        'probability_loss': float(proba_derrota),
        'prediction': pred_resultado,
        'actual': real_resultado,
        'correct': bool(real_resultado == pred_resultado)
    })

# --- 8. SALVAR RESULTADOS EM JSON ---
results = {
    'model_performance': {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    },
    'coefficients': {
        'intercept': float(intercepto),
        'features': [{'feature': feature, 'coefficient': float(model_log.coef_[0][i])}
                    for i, feature in enumerate(FEATURES_LOG)]
    },
    'correlations': target_correlation_log.drop(TARGET_LOG, errors='ignore').to_dict(),
    'feature_names': FEATURES_LOG,
    'equation': equation,
    'examples': examples_list
}

# Salvar em arquivo JSON
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(output_dir, 'logistic_regression_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Resultados salvos em: {output_file}")
print("\n" + "="*80)
print("ANALISE CONCLUIDA")
print("="*80)