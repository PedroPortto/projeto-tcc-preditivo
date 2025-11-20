# src/models/model_final.py

import pandas as pd
from datetime import timedelta 
from src.models.optimize_ml import load_and_prepare_data, create_continuous_series, optimize_and_forecast
from typing import List
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

# Definição de Caminhos
FACT_TABLE_PATH = 'data/processed/daily_fact_table.parquet'
# Constante Global (Referência)
DEFAULT_POWERBI_PATH = 'data/processed/powerbi_dataset_final.parquet'

# Regras de Negócio: Horizontes de previsão (RN05)
FORECAST_HORIZONS = [7, 14, 30] # dias

def generate_multi_horizon_forecast(df_fact: pd.DataFrame, horizons: List[int]):
    
    GROUP_COLS = ['normalized_category', 'entities_id']
    df_series_continuous = create_continuous_series(df_fact, GROUP_COLS)
    
    all_forecasts_final = []
    groups = df_series_continuous.groupby(GROUP_COLS)
    
    for name, group_df in groups:
        category, entity = name
        if len(group_df) < 30: continue 

        print(f"Gerando previsões para {category}/{entity}...")

        try:
            # Recebe os 3 valores retornados pelo optimize_ml
            df_forecast_30, mape_30, _ = optimize_and_forecast(group_df, horizon=30) 
            
            start_date_forecast = df_forecast_30['date'].min()
            
            for h in horizons:
                end_date_forecast = start_date_forecast + timedelta(days=h - 1)
                
                df_future_forecast = df_forecast_30.loc[
                    (df_forecast_30['date'] >= start_date_forecast) & 
                    (df_forecast_30['date'] <= end_date_forecast)
                ].copy()
                
                df_future_forecast = df_future_forecast.assign(
                    normalized_category=category,
                    entities_id=entity,
                    horizon=h
                )
                all_forecasts_final.append(df_future_forecast[['date', 'normalized_category', 'entities_id', 'horizon', 'P50_volume', 'P90_volume']])

        except Exception as e:
            print(f"Erro ao gerar previsão: {e}")
    
    if not all_forecasts_final:
        print("Nenhuma previsão gerada.")
        return
        
    df_forecasts = pd.concat(all_forecasts_final)
    
    # Histórico
    df_fact['horizon'] = 0 
    df_fact['P50_volume'] = df_fact['volume']
    df_fact['P90_volume'] = df_fact['volume']
    
    df_history_dataset = df_fact.rename(columns={'volume': 'volume_real'}).copy()
    df_forecast_dataset = df_forecasts.assign(volume_real=pd.NA).copy()
    
    # Unificação
    df_final_dataset = pd.concat([
        df_history_dataset[['date', 'normalized_category', 'entities_id', 'horizon', 'volume_real', 'avg_ttr_hours', 'P50_volume', 'P90_volume']],
        df_forecast_dataset.rename(columns={'P50_volume': 'P50_volume', 'P90_volume': 'P90_volume'})
        [['date', 'normalized_category', 'entities_id', 'horizon', 'volume_real', 'P50_volume', 'P90_volume']]
    ], ignore_index=True)
    
    df_final_dataset['avg_ttr_hours'] = df_final_dataset['avg_ttr_hours'].fillna(method='ffill')

    # 🚨 TRATAMENTO FINAL (Inteiros e Limpeza)
    MAX_SAFE_VOLUME = 5000.0 

    # Colunas que DEVEM ser inteiros (P50 e P90)
    for col in ['P50_volume', 'P90_volume']:
        df_final_dataset[col] = pd.to_numeric(df_final_dataset[col], errors='coerce')
        df_final_dataset[col] = df_final_dataset[col].fillna(0) 
        df_final_dataset[col] = df_final_dataset[col].clip(upper=MAX_SAFE_VOLUME)
        df_final_dataset[col] = df_final_dataset[col].round(0).astype(int) # Conversão para INT
        
    # Volume Real
    df_final_dataset['volume_real'] = pd.to_numeric(df_final_dataset['volume_real'], errors='coerce')
    df_final_dataset['volume_real'] = df_final_dataset['volume_real'].clip(upper=MAX_SAFE_VOLUME)

    # --- LÓGICA DE SALVAMENTO SEGURA ---
    # Usa uma variável LOCAL 'save_path' para não confundir com a global
    save_path = DEFAULT_POWERBI_PATH

    if os.path.exists(save_path):
        try:
            os.remove(save_path)
            print("Arquivo antigo removido.")
        except PermissionError:
            print("⚠️ FECHE O POWER BI! Salvando em _v2...")
            save_path = 'data/processed/powerbi_dataset_final_v2.parquet'

    df_final_dataset.to_parquet(save_path, index=False)
    
    print("\n--- MODELO FINAL PRONTO (INT) ---")
    print(f"Dataset salvo em: {save_path}")
    # Pega o último valor válido para mostrar no print (evita erro se estiver vazio)
    if not df_final_dataset.empty:
         print(f"Exemplo de valor P90 (deve ser inteiro): {df_final_dataset['P90_volume'].iloc[-1]}")

if __name__ == "__main__":
    df_fact_table = load_and_prepare_data(FACT_TABLE_PATH)
    generate_multi_horizon_forecast(df_fact_table, FORECAST_HORIZONS)