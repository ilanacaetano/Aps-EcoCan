# EcoCan: Tecnologia em Prol da Reciclagem Sustentável

## ♻️ Visão Geral do Projeto

Este projeto, desenvolvido como parte da disciplina de Atividades Práticas Supervisionadas (APS) de Processamento de Imagens e Visão Computacional, visa aplicar a tecnologia para resolver um problema real de sustentabilidade ambiental: a classificação automática de materiais recicláveis.

O **EcoCan** é uma iniciativa que atua no desenvolvimento de um programa capaz de identificar automaticamente latas de alumínio por meio de técnicas avançadas de Processamento de Imagens e Visão Computacional.

## 🧠 Problema e Motivação

Apesar de o Brasil apresentar altas taxas de reciclagem de latas de alumínio (cerca de 97,3% em 2024, após um pico de 100,1% em 2023), a **separação de materiais metálicos** de outros resíduos ainda depende majoritariamente de mão de obra manual.

A triagem manual é caracterizada por ser:

  * Lenta e sujeita a erros.
  * Nem sempre segura ou bem remunerada.
  * Sujeita a desafios operacionais, de coleta ou de logística de sucata.

A nossa solução busca otimizar esse processo, proporcionando uma ferramenta de **reconhecimento inteligente** capaz de aumentar a acurácia da separação e reduzir rejeitos ou contaminação.

## ⚙️ Solução Técnica

O diferencial do EcoCan está no desenvolvimento de um algoritmo de reconhecimento inteligente, que utiliza uma combinação de filtros clássicos de Processamento Digital de Imagens (PDI) e um modelo de *Machine Learning* para classificar objetos como "LATA" ou "OUTRO".

### Pipeline de Processamento de Imagens

O processo de classificação envolve as seguintes etapas:

1.  **Pré-processamento e Filtragem:** Prepara a imagem, focando em:
      * **Redução de Ruído:** Utilizando filtros de Média para evitar a detecção de bordas falsas.
      * **Realce de Bordas e Destaque da Forma Cilíndrica:** Uso de filtros como o Sobel X e Sobel Y para identificar contornos horizontais e verticais , e o filtro Sharpen para realçar a forma do objeto.
      * **Captura de Detalhes Finos:** Aplicação do filtro Laplaciano.
        
2.  **Segmentação:** Separação do objeto (lata) do fundo, destacando o formato cilíndrico e as bordas. Isso é complementado pela detecção de bordas Canny.

3.  **Extração de Características (Features):** As características utilizadas para treinar o modelo de IA são baseadas nas estatísticas (média e desvio padrão) das imagens após a aplicação dos filtros.

4.  **Reconhecimento (IA):** Treinamento de um modelo de Inteligência Artificial para a classificação binária. O modelo utilizado é uma **Support Vector Machine (SVM)** com *kernel* linear (`SVC(kernel='linear', C=1.0)`).

### 💻 Tecnologias

  * **Linguagem:** Python
  * **Visão Computacional:** OpenCV (`cv2`)
  * **Machine Learning:** Scikit-learn (`sklearn`), Joblib (para persistência do modelo)
  * **Interface Gráfica (GUI):** Ttkbootstrap (baseado em Tkinter)
  * **Manipulação de Imagens:** NumPy, PIL/Pillow

## 📂 Estrutura do Repositório

```
Aps-EcoCan/
├── Projeto/
│   ├── data/
│   │   ├── latas/          # Imagens de latas (o que deve ser classificado)
│   │   └── outros/         # Imagens de outros resíduos (o que não deve ser classificado)
│   ├── main.py             # Código fonte principal da aplicação (GUI, Treinamento, Classificação)
│   ├── Ecocan.png          # Logo do Projeto
│   └── lata_teste.jpg      # Exemplo de imagem para teste
├── resultados/             # Imagens resultantes da aplicação de filtros (ex: classificada_bordas.jpg, classificada_sobel.jpg)
├── Ecocan.pdf              # Apresentação Teórica do Projeto
├── modelo_latas_aug.pkl    # Modelo de IA (SVC + StandardScaler) treinado e serializado
├── README.md               # Este arquivo
└── LICENSE                 # Informações da licença (MIT)
```

## ▶️ Como Executar o Projeto

### Pré-requisitos

Certifique-se de ter o Python instalado (o projeto foi desenvolvido com uma versão Python 3.x) e as seguintes bibliotecas:

```bash
pip install opencv-python scikit-learn joblib numpy ttkbootstrap pillow
```

### Uso da Aplicação

1.  **Execute a aplicação GUI:**
    ```bash
    python Projeto/main.py
    ```

2.  **Treine o Modelo:**
      * Clique no botão **"Treinar Modelo"**. O script irá carregar as imagens de `Projeto/data/latas` e `Projeto/data/outros`, aplicar as técnicas de pré-processamento, realizar a extração de *features* (incluindo *data augmentation* para robustez), e treinar o modelo SVM, salvando-o como `modelo_latas_aug.pkl`.

3.  **Classifique uma Imagem:**
      * Clique no botão **"Selecionar Imagem"** e escolha um arquivo de imagem (`.jpg`, `.png`, `.jpeg`).
      * O sistema irá carregar a imagem, extrair suas características e utilizar o modelo treinado para prever o resultado, exibindo **"LATA"** ou **"OUTRO"** na interface.

4.  **Visualize os Filtros:**
      * O botão **"Ver Filtros"** abre as janelas de visualização dos resultados dos filtros aplicados na última imagem classificada/testada, que são salvos na pasta `resultados/`.

## 👥 Autores

O projeto foi desenvolvido pelo grupo de alunos:

  * Amanda Eleoterio Silva (RA: N9514H5)
  * Erik Alves Gonçalves (RA: N2246A3) 
  * Ilana dos Santos Caetano da Silva (RA: N0810CO) 
  * Maria Luiza dos Anjos Santos (RA: F353478) 
  * Isabelly Cristina Araujo (RA: G514572) 

## 📜 Licença

Este projeto está licenciado sob a **Licença MIT**.

**Copyright (c) 2025 Lana ⋆˚࿔**

Consulte o arquivo [LICENSE](https://www.google.com/search?q=ilanacaetano/aps-ecocan/Aps-EcoCan-82b8f6e754b22c206ecbbb6cfc4f571bf7ff63bf/LICENSE) para mais detalhes.
