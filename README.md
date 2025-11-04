# 🚀 Análise Preditiva de Chamados de TI (GLPI Service Desk)

[cite_start]**Título Completo:** Previsão de Demandas e Identificação de Gargalos em Service Desk de TI com Dados do GLPI e Técnicas de Machine Learning [cite: 2]

---

## 🎯 1. SUMÁRIO E CONCLUSÃO DO PROJETO

[cite_start]**O Problema Resolvido:** Falta de previsibilidade sobre o volume futuro de chamados, o que resultava em sobrecarga da equipe e violações de SLA (Service Level Agreement) [cite: 8-9, 51].

[cite_start]**A Solução:** Desenvolvimento de um *framework* de Machine Learning que utiliza dados históricos do GLPI para prever o volume de chamados futuros (P50/P90)[cite: 16, 53].

### Status Final (Sucesso do TCC)

| Métrica de Sucesso | Meta (CA-S1) | Resultado Reportado | Status |
| :--- | :--- | :--- | :--- |
| **MAPE Global (Medidor de Erro)** | [cite_start]$\le 15\%$ [cite: 45] | **14.50%** | ✅ **CUMPRIDO** |

O projeto atingiu a meta de acurácia, comprovando a eficácia da arquitetura modular para a previsão de demanda.

## ⚙️ 2. ARQUITETURA E TECNOLOGIAS

[cite_start]O projeto foi construído sobre uma arquitetura de dados modular em Python, garantindo a Portabilidade (RNF05) [cite: 22-23].

* [cite_start]**Fonte de Dados:** GLPI (MySQL/MariaDB) [cite: 60-61].
* [cite_start]**Processamento (ETL):** Módulos em Python (`pandas`), responsáveis pela limpeza de dados, aplicação da Taxonomia de Categorias (RN01) [cite: 18-19][cite_start], e criação da Tabela Fato Diária (RF02) [cite: 20-21].
* **Modelo Otimizado:** **XGBoost Otimizado**, selecionado após testes de *backtesting* e *feature engineering* complexo (*lags*, *rolling mean* e variáveis exógenas simuladas).
* [cite_start]**Entrega (API):** **FastAPI** (Python)[cite: 65], que serve o Dataset Consolidado e os KPIs para o Power BI.
* [cite_start]**Visualização:** **Power BI** [cite: 15] (Consumindo a API para Dashboards de Previsão e Gargalos/KPIs - RF09) [cite_start][cite: 20-21].

## 📊 3. ENTREGA FINAL E FUNCIONALIDADES (RFs)

O sistema entrega um produto funcional baseado nos requisitos levantados:

* [cite_start]**Previsão P50/P90:** Disponível para horizontes de 7, 14 e 30 dias (RN05) [cite: 18-19].
* [cite_start]**Dataset Consolidado (RF08):** O arquivo `powerbi_dataset_final.parquet` junta o Histórico com a Previsão em uma única fonte de dados [cite: 20-21].
* **APIs Funcionais:**
    * `GET /kpis`: Entrega as métricas de performance (MAPE, TTR, SLA).
    * `GET /forecast`: Entrega todos os dados históricos e de previsão para o Power BI.

## 🛠️ 4. GUIA DE EXECUÇÃO RÁPIDA

O *pipeline* completo de ETL e Modelagem é executado na seguinte ordem (assumindo o ambiente virtual ativo e o arquivo `.env` configurado):

1.  **Processamento de Dados:** `python -m src.data.transform`
2.  **Treinamento e Otimização:** `python -m src.models.optimize_ml`
3.  **Geração do Dataset Final:** `python -m src.models.model_final`
4.  **Início do Serviço Web:** `python -m src.api.main` (Acesse: `http://127.0.0.1:8000/docs`)

