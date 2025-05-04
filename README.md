# UFRGS - INF01017 - Aprendizado de Máquina

Esse repositório contém os trabalhos práticos da disciplina de INF01017 - Aprendizado de Máquina.

## Como executar?

O projeto utiliza o [Poetry](https://python-poetry.org/) para gerenciamento de dependências e ambientes virtuais. Certifique-se de ter o Poetry instalado antes de começar.

### Passo a passo:

1. Clone o repositório:

    ```bash
    git clone https://github.com/seu-usuario/ML-UFRGS-2025-1.git
    cd ML-UFRGS-2025-1
    ```

2. Instale as dependências com o Poetry:
    ```bash
    poetry install
    ```
3. Execute o projeto:
    ```bash
    poetry run python src/main.py
    ```
# Estrutura do projeto
A estrutura do projeto está organizada da seguinte forma: 

    .
    ├── data/                     # Dados utilizados no projeto
    ├── graphs/                   # Gráficos gerados ao longo do projeto
    ├── src/                      # Código-fonte do projeto
    |   └── models/
    |       └── knn.py            # Treina o modelo KNN.
    |   └── main.py               # Script principal de execução
    │   └── data_processing.py    # Realiza o pré-processamento dos dados
    ├── pyproject.toml            # Configuração do Poetry
    └── README.md
  
# Referências
Durante o desenvolvimento deste trabalho, utilizamos o seguinte dataset como base para o treinamento e avaliação dos modelos de machine learning: [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?resource=download)