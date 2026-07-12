# 📚 NLP Review Intelligence

Sistema de análise inteligente de avaliações de livros utilizando NLP (Natural Language Processing) para extrair insights sobre aspectos narrativos, estilo de escrita e ritmo de leitura. Permite identificar usuários ideais para entrevistas (promoters e detractors) com base em análise semântica de reviews.

---

## 📋 Visão Geral do Projeto

Este projeto processa dados de avaliações de livros e realiza:
- **Limpeza e Normalização** de dados textuais e categóricos
- **Análise de Sentimento** com métricas específicas (plot, style, pacing)
- **Vetorização** de reviews para busca semântica
- **Agregação de Métricas** por autor e gênero
- **Dashboard Interativo** (Streamlit) para exploração dos resultados

---

## 🔧 Pré-requisitos

- **Python 3.12+**
- **pip** ou **uv** (gerenciador de pacotes)
- ~5GB de espaço em disco (para cache de modelos e dados processados)

---

## 📦 Instalação

### 1️⃣ Clonar o Repositório

```bash
git clone <seu-repositorio>
cd Case-A3
```

### 2️⃣ Instalar Dependências

#### Opção A: Usando `uv` (recomendado - mais rápido)

```bash
uv sync
```

#### Opção B: Usando `pip` com `pyproject.toml`

```bash
pip install -e .
```

Ou instalar manualmente:

```bash
pip install polars pandas plotly streamlit sentence-transformers nltk faiss-cpu onnxruntime-directml openai pyarrow tiktoken wordcloud text-unidecode tqdm-joblib chromadb
```

### 3️⃣ Verificar Instalação

```bash
python -c "import polars; import streamlit; print('✅ Dependências instaladas com sucesso!')"
```

---

## 📊 Dados de Entrada

Coloque os arquivos CSV no diretório `data/`:

- **`data/books_data.csv`** - Informações dos livros (Title, authors, categories, summary, etc)
- **`data/Books_rating.csv`** - Avaliações e reviews (Title, User_id, score, text, etc)

### Estrutura esperada:

```
data/
├── books_data.csv          # Dados dos livros
├── Books_rating.csv        # Avaliações dos usuários
└── [arquivos processados]  # Gerados durante execução
```

---

## ▶️ Como Executar

### 🚀 Fluxo Completo de Processamento

O projeto possui uma **ordem de execução específica**. Execute os passos na seguinte sequência:

#### **Passo 1: Limpeza e Normalização de Dados**

```bash
cd core
python main.py
```

**O que acontece:**
1. `cleaning.py` - Limpa textos, normaliza autores, mapeia títulos
2. `genre_normalize.py` - Normaliza categorias/gêneros
3. `review_analysis.py` - Análise de sentimento e aspectos NLP

**Saída gerada:**
- `data/cleaned_reviews.parquet`
- `data/cleaned_reviews_mapped.parquet`
- `data/data_sentiment.parquet`
- `vector_db/reviews_index.faiss` (índice FAISS para busca semântica)

**Tempo estimado:** 30-60 minutos (depende do tamanho dos dados e hardware)

#### **Passo 2: Preparar Dados para Dashboard**

```bash
python app/prepare_data.py
```

**O que acontece:**
- Carrega `data_sentiment.parquet`
- Remove coluna pesada de embeddings
- Agregação de métricas por autor e gênero
- Calcula satisfação, polarização e dimensões NLP

**Saída gerada:**
- `data/author_metrics.parquet`
- `data/genre_metrics.parquet`
- `data/reviews_interview.parquet`

**Tempo estimado:** 5-10 minutos

#### **Passo 3: Iniciar Dashboard Interativo**

```bash
streamlit run app/app.py
```

**Saída esperada:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Abra a URL no navegador e comece a explorar os dados!

---

## 📁 Estrutura de Arquivos

```
Case-A3/
│
├── 📄 README.md                    # Este arquivo
├── 📄 pyproject.toml              # Configuração de dependências
├── 📓 EDA_analysis.ipynb          # Notebooks de análise exploratória
│
├── 📂 data/                        # Dados (entrada e saída)
│   ├── books_data.csv             # [INPUT] Dados dos livros
│   ├── Books_rating.csv           # [INPUT] Avaliações
│   ├── cleaned_reviews.parquet    # [OUTPUT - Passo 1]
│   ├── cleaned_reviews_mapped.parquet  # [OUTPUT - Passo 1]
│   ├── data_sentiment.parquet     # [OUTPUT - Passo 1]
│   ├── author_metrics.parquet     # [OUTPUT - Passo 2]
│   ├── genre_metrics.parquet      # [OUTPUT - Passo 2]
│   └── reviews_interview.parquet  # [OUTPUT - Passo 2]
│
├── 📂 core/                        # Scripts de processamento
│   ├── main.py                    # Orquestrador principal
│   ├── cleaning.py                # Limpeza de dados
│   ├── genre_normalize.py         # Normalização de gêneros
│   └── review_analysis.py         # Análise NLP e sentimento
│
├── 📂 app/                         # Aplicação Streamlit
│   ├── app.py                     # Dashboard interativo
│   ├── prepare_data.py            # Preparação de dados para dashboard
│   └── build_faiss.py             # [OPCIONAL] Reconstruir índice FAISS
│
├── 📂 presentation/               # Apresentação
│   └── build_presentation.py      # Script para gerar apresentação
│
├── 📂 hf_cache/                   # Cache de modelos Hugging Face
│   └── models--sentence-transformers--all-MiniLM-L6-v2/  # Modelo pré-treinado
│
└── 📂 vector_db/                  # Banco de dados vetorial
    └── reviews_index.faiss        # Índice FAISS para busca semântica
```

---

## 🔄 Detalhamento dos Passos de Processamento

### **Passo 1: `core/main.py`**

Executa três módulos de processamento em sequência:

#### 1.1 `cleaning.py`
- ✅ Carrega CSVs (books_data.csv, Books_rating.csv)
- ✅ Join entre livros e avaliações
- ✅ Limpeza de texto (remove stopwords, caracteres especiais)
- ✅ Normalização de autores (padronização de nomes)
- ✅ Tratamento de valores nulos
- 📤 Output: `cleaned_reviews.parquet`

#### 1.2 `genre_normalize.py`
- ✅ Carrega dados limpos
- ✅ Normaliza categorias/gêneros usando IA
- ✅ Mapeamento de categorias complexas para gêneros principais
- 📤 Output: `cleaned_reviews_mapped.parquet`

#### 1.3 `review_analysis.py`
- ✅ Análise de Sentimento (VADER + Transformer)
- ✅ Extração de Aspectos NLP:
  - **Plot**: história, desenvolvimento, narrativa
  - **Style**: escrita, prosa, qualidade da linguagem
  - **Pacing**: ritmo, velocidade, fluxo da leitura
- ✅ Vetorização de reviews (Sentence Transformers)
- ✅ Construção de índice FAISS para busca semântica
- 📤 Output: `data_sentiment.parquet`, `vector_db/reviews_index.faiss`

### **Passo 2: `app/prepare_data.py`**

Processa dados de sentimento para o dashboard:
- ✅ Agregação por Autor (reviews count, avg_score, satisfaction_rate, polarization, plot, style, pacing)
- ✅ Agregação por Gênero (mesmas métricas)
- ✅ Seleção de reviews para entrevista (promoters e detractors)
- 📤 Output: 3 arquivos Parquet otimizados para Streamlit

### **Passo 3: `app/app.py`**

Dashboard interativo com:
- 📊 Análise de Performance (por Autor ou Gênero)
- 📈 Visualizações de Dimensões NLP (Plot, Style, Pacing)
- 🎤 Seleção de usuários para entrevista (Promoters e Detractors)
- 💬 Exibição de reviews diretos do usuário

---

## 🆘 Troubleshooting

### ❌ Erro: "No module named 'polars'"
```bash
pip install polars
```

### ❌ Erro: "FAISS index not found"
Reconstrua o índice:
```bash
python app/build_faiss.py
```

### ❌ Streamlit não inicia
```bash
streamlit run app/app.py --logger.level=debug
```

### ❌ Procesamento lento / memória insuficiente
- Reduza o tamanho dos dados de entrada
- Execute em máquina com mais RAM (16GB+ recomendado)
- Use `polars` em vez de `pandas` (já fazemos isso)

### ❌ Modelos NLP não baixam
Verifique conexão com internet e espaço em disco:
```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## 📝 Notas Importantes

1. **Primeira Execução**: A primeira vez que você roda `core/main.py`, os modelos NLP são baixados automaticamente (~500MB)
2. **Cache**: Os modelos ficam em `hf_cache/` para reuso posterior
3. **GPU**: Por padrão usa CPU. Para GPU (CUDA), ajuste `core/review_analysis.py`
4. **Dados Sensíveis**: Não commit arquivos CSV com dados reais no git

---

## 🚀 Próximos Passos (TODO)

- [ ] Efetivar vetorização com GPU (CUDA)
- [ ] Filtro por Tópico Específico: barra de pesquisa semântica para recrutar usuários
- [ ] Integração com LLM para análise mais profunda de reviews
- [ ] API REST para integração externa

---

## 📧 Suporte

Para dúvidas ou problemas, verifique:
- Arquivo de log do Streamlit
- Output do terminal durante `core/main.py`
- Tamanho dos arquivos em `data/`

---

**Criado com ❤️ para análise inteligente de reviews**