#  Classificação de Doenças Neurodegenerativas Utilizando EEG

**Projeto desenvolvido na disciplina MO444 - Aprendizado de Máquina, na Unicamp.**  
O objetivo foi classificar doenças neurodegenerativas, como Alzheimer (AD) e Demência Frontotemporal (FTD), a partir de sinais de EEG de pacientes. Duas abordagens foram testadas:

- **Aprendizado Auto-Supervisionado**: Utilizando Contrastive Predictive Coding (CPC) para extração de representações dos sinais de EEG.
- **Aprendizado Supervisionado** (*Minha principal contribuição*): Desenvolvimento e otimização de um modelo híbrido CNN-LSTM para classificação.

---

##  Objetivos

1. **Classificação de EEG**: Diferenciar sinais de pacientes saudáveis de pacientes com doenças neurodegenerativas.
2. **Exploração de Técnicas de Aprendizado de Máquina**:
   - Extração de representações latentes via CPC.
   - Uso de **CNN+LSTM** para modelagem supervisionada dos sinais temporais.
   - Análise de hiperparâmetros via **Optuna** para otimização do modelo.

---

##  Metodologia

### **1️ Pré-Processamento dos Dados**
- **Sinais de EEG** de 88 indivíduos (19 eletrodos, taxa de amostragem de 500 Hz).
- **Filtragem** de sinais entre 0.5 e 45 Hz.
- **Segmentação** em janelas de 1 segundo (256 timestamps por janela).
- **Conversão** para formato `.npy` para compatibilidade com bibliotecas de ML.

### **2️ Modelagem Supervisionada - Minha Principal Contribuição**
> Desenvolvimento e otimização do modelo CNN-LSTM para análise dos sinais de EEG.

- **Arquitetura CNN+LSTM**:
  - **CNN 1D** para extração de padrões espaciais nos sinais EEG.
  - **LSTM** para modelagem de dependências temporais nos sinais.
  - **Dropout e Regularização L2** para mitigar overfitting.

- **Treinamento**:
  - Otimização de hiperparâmetros com **Optuna**.
  - **Early Stopping** para evitar sobreajuste.
  - **Threshold tuning** para priorizar recall (minimizando falsos negativos em diagnósticos).

### **3️ Comparação com Benchmark**
O modelo supervisionado foi comparado com um benchmark baseado em Random Forests e MLP, utilizado no artigo original do dataset. Os resultados indicaram desafios no aprendizado profundo devido ao sobreajuste.

---

##  Resultados


### **📌 Comparação com o Benchmark**
| Modelo                | Acurácia | Precisão | Recall | F1-Score | AUC  |
|---------------------|---------|----------|--------|----------|------|
| **CNN-LSTM (meu modelo)** | 61%     | 63%      | 59%    | 57%      | 73%  |
| **Benchmark (Random Forests/MLP)** | 81%     | 57%      | 78%    | 66%      | -    |

Apesar das dificuldades enfrentadas, conseguimos **otimizar recall**, priorizando diagnósticos sensíveis.

---

##  Conclusões e Melhorias Futuras

1. **Explorar Tamanhos de Janela Maiores**: Testar janelas de 512 timestamps para capturar padrões mais amplos nos sinais de EEG.
2. **Testar Outras Estratégias de Regularização**: Como batch normalization e dropout mais agressivo.
3. **Analisar a Importância dos Eletrodos**: Determinar quais sensores do EEG contribuem mais para o diagnóstico.
4. **Explorar Diferentes Estratégias de Pré-Treinamento**: Considerar tarefas pretexto mais adequadas ao domínio médico.

---

##  Habilidades Demonstradas

-  **Processamento e Análise de Sinais** (EEG, séries temporais)
-  **Aprendizado de Máquina Supervisionado** (CNN, LSTM)
-  **Otimização de Modelos** (Optuna, tuning de hiperparâmetros)
-  **Manipulação de Dados com Python** (`numpy`, `pandas`, `sklearn`, `tensorflow`)
-  **Visualização e Interpretação de Resultados** (`matplotlib`, `seaborn`, `t-SNE`)

---


---

## 📚 Referências

- Miltiadous, A. et al. (2023). *A dataset of EEG recordings from Alzheimer’s disease, Frontotemporal dementia and Healthy subjects*.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation.
- LeCun, Y. et al. (1995). *Convolutional networks for images, speech, and time series*.
