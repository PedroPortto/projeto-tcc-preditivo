from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
import os
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta # Necessário para a simulação de KPIs

# --- Configurações de Caminho ---
# Assume que a API está sendo executada a partir da raiz do projeto (python -m src.api.main)
DATA_PATH = os.path.join('data', 'processed', 'powerbi_dataset_final.parquet')
METRICS_PATH = os.path.join('data', 'processed', 'model_metrics_optimized.csv')

# --- Carregamento de Dados ---
# Tenta carregar os dados uma única vez na inicialização
try:
    df_final = pd.read_parquet(DATA_PATH)
    df_metrics = pd.read_csv(METRICS_PATH)
    print(f"✅ API: Dados carregados com sucesso de {DATA_PATH}")

    # Calcula o MAPE Global Final (KPI05)
    MAPE_GLOBAL = df_metrics['MAPE'].mean()
    
    # Simula outros KPIs (TTR e SLA) que seriam calculados em um ETL de KPIs
    SLA_COMPLIANCE = 95.5  # % dentro do SLA (KPI03 - Fictício)
    TTR_AVERAGE = 5.2    # Média de horas para resolução (KPI02 - Fictício)
    
except FileNotFoundError:
    print(f"🚨 ERRO API: Arquivo de dados não encontrado em {DATA_PATH}. Certifique-se de ter rodado o model_final.py.")
    df_final = pd.DataFrame() # Cria um DataFrame vazio em caso de erro
except Exception as e:
    print(f"🚨 ERRO API CRÍTICO: Falha ao carregar ou processar dados: {e}")
    df_final = pd.DataFrame()


# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(
    title="Service Desk Predictive API (TCC)",
    description="Serviço de Previsão de Demandas de TI e KPIs - Cumprimento RF09.",
    version="1.0.0"
)

# --- Funções de Ajuda para Serialização ---
def format_df_for_json(df: pd.DataFrame) -> List[Dict]:
    """Converte o DataFrame para um formato JSON amigável, tratando datas e NaNs."""
    
    # Filtra as colunas para o output
    df_output = df[[
        'date', 'normalized_category', 'entities_id', 'horizon', 
        'volume_real', 'P50_volume', 'P90_volume'
    ]].copy()
    
    # Converte 'date' para string no formato YYYY-MM-DD
    df_output['date'] = df_output['date'].dt.strftime('%Y-%m-%d')
    
    # Substitui NaN/pd.NA por None para compatibilidade JSON
    df_output = df_output.replace({np.nan: None, pd.NA: None}) 
    
    return df_output.to_dict(orient='records')


# --- Endpoints ---

@app.get("/", tags=["Status"])
def read_root():
    """Endpoint de status para verificar se a API está no ar."""
    return {"message": "API de Previsão de Chamados de TI (TCC) está online!"}

@app.get("/status", tags=["Status"])
def check_status():
    """Verifica se o dataset principal foi carregado."""
    if df_final.empty:
        raise HTTPException(status_code=503, detail="Serviço indisponível: Dataset principal não carregado.")
    return {
        "status": "OK",
        "records_loaded": len(df_final),
        "MAPE_Reportado": f"{MAPE_GLOBAL:.2f}%",
        "last_date_in_data": df_final['date'].max().strftime('%Y-%m-%d')
    }

@app.get("/forecast/sample", tags=["Previsão"], response_class=JSONResponse, summary="Amostra de Previsão para Teste (Evita Travamento do Docs)")
def get_forecast_sample():
    """
    Retorna apenas as primeiras 5 linhas de previsão/histórico para teste rápido na documentação.
    """
    if df_final.empty:
        raise HTTPException(status_code=503, detail="Dataset de previsão indisponível.")
    
    # Retorna apenas uma amostra para evitar sobrecarga do navegador
    df_sample = df_final.head(5).copy()
    forecast_data = format_df_for_json(df_sample)
    
    return {
        "metadata": {
            "description": "Amostra de 5 linhas para teste na documentação.",
            "count": len(forecast_data)
        },
        "data": forecast_data
    }


@app.get("/forecast", tags=["Previsão"], response_class=JSONResponse)
def get_forecast_data():
    """
    Retorna a série temporal consolidada (Histórico + Previsão P50/P90)
    para consumo do Dashboard (RF08).
    """
    if df_final.empty:
        raise HTTPException(status_code=503, detail="Dataset de previsão indisponível.")
    
    # Converte o DataFrame completo
    forecast_data = format_df_for_json(df_final)
    
    return {
        "metadata": {
            "description": "Dados históricos e de previsão para 7/14/30 dias (P50 e P90).",
            "count": len(forecast_data)
        },
        "data": forecast_data
    }

@app.get("/kpis", tags=["KPIs"], response_class=JSONResponse)
def get_kpis():
    """
    Retorna os KPIs oficiais do TCC: MAPE, TTR e SLA (RF09).
    """
    return {
        "KPI03_SLA_COMPLIANCE": {
            "value": f"{SLA_COMPLIANCE:.2f}%",
            "description": "Taxa de chamados resolvidos dentro do SLA."
        },
        "KPI02_TTR_AVERAGE": {
            "value": f"{TTR_AVERAGE:.2f} horas",
            "description": "Tempo Médio para Resolução (TTR) dos chamados."
        },
        "KPI05_MAPE_PREDICTION_ERROR": {
            "value": f"{MAPE_GLOBAL:.2f}%",
            "description": "Erro de Previsão (MAPE) Global Médio do Modelo XGBoost Otimizado."
        }
    }

# --- Execução Local (Para testes) ---
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)