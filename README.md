

# 🏥 Projeto: Previsão de Custos de Seguro de Saúde

Este projeto de Machine Learning tem como objetivo prever o valor dos custos de seguro de saúde para um indivíduo com base em suas características pessoais. Para isso, foi desenvolvido um pipeline completo de regressão e uma aplicação web interativa.

## 1. Análise e Preparação dos Dados 📊

A primeira etapa do projeto foi realizada no notebook `model.ipynb`, utilizando o conjunto de dados `insurance.csv`.

### Carregamento e Exploração (EDA)
* **Carregamento:** Os dados foram carregados e estruturados em um DataFrame do Pandas.
* **Análise de Distribuição:** Foi feita uma análise estatística para entender o comportamento de variáveis como `age` (idade) e `bmi` (IMC) em relação à variável alvo `charges` (custos).
* **Visualização:** Foram utilizados histogramas, boxplots e gráficos de dispersão para identificar tendências.
    * *Insight:* Identificou-se uma forte correlação positiva entre ser fumante (`smoker`) e ter custos de seguro mais elevados.

### Pré-processamento dos Dados
Para garantir a qualidade do modelo, os dados passaram por transformações rigorosas:

* **Codificação de Variáveis Categóricas:**
    * Variáveis de texto como `gender`, `diabetic` e `smoker` foram convertidas para formato numérico utilizando **LabelEncoder**.
    * Exemplo: Categorias como 'male'/'female' foram transformadas em 0 e 1.
    * *Artefatos:* Os codificadores foram salvos em arquivos `label_encoder_*.pkl` para reutilização.
* **Normalização de Variáveis Numéricas:**
    * Variáveis com escalas distintas (`age`, `bmi`, `bloodpressure`) foram ajustadas utilizando **StandardScaler**.
    * O scaler padroniza os dados para média 0 e desvio padrão 1, evitando viés no modelo.
    * *Artefato:* O normalizador foi salvo no arquivo `scaler.pkl`.

---

## 2. Construção e Treinamento do Modelo 🤖

Com os dados processados, iniciou-se a fase de modelagem preditiva.

* **Divisão dos Dados:** O dataset foi separado em grupos de **treino** (aprendizado) e **teste** (validação de performance).
* **Seleção do Modelo:** Após testes com diferentes algoritmos, o modelo escolhido foi o **XGBoost Regressor**, devido à sua alta capacidade de generalização e performance em dados tabulares.
* **Avaliação:** O modelo foi avaliado utilizando métricas como o **R² (Coeficiente de Determinação)**, indicando o quanto as características explicam a variação dos custos.
* **Persistência do Modelo:** O melhor modelo treinado foi salvo no arquivo `best_model.pkl` utilizando a biblioteca **Joblib**, permitindo o uso posterior sem necessidade de retreinamento.

---

## 3. Criação da Aplicação Web 🌐

A etapa final consistiu em desenvolver uma interface amigável para interação com o usuário final, contida no arquivo `app.py`.

* **Ferramenta:** A aplicação foi construída utilizando **Streamlit**.
* **Fluxo de Execução:**
    1.  **Inicialização:** O app carrega os artefatos (`best_model.pkl`, `scaler.pkl`, `label_encoder_*.pkl`).
    2.  **Input do Usuário:** Interface com formulário para inserção de dados (Idade, IMC, Gênero, etc.).
    3.  **Processamento e Predição:**
        * Ao clicar em *"Predict Payment"*, os dados são coletados em um DataFrame.
        * Aplica-se o **mesmo pré-processamento** do treino (Encoding e Scaling) usando os artefatos carregados.
        * O modelo realiza a inferência através do método `.predict()`.
    4.  **Resultado:** O valor previsto do seguro é exibido na tela de forma clara.

---

