import pandas as pd
# Importação de timedelta corrigida
from datetime import timedelta 
# Reutilizando funções chave dos módulos anteriores
from src.models.optimize_ml import load_and_prepare_data, create_continuous_series, optimize_and_forecast
from typing import List
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Definição de Caminhos
FACT_TABLE_PATH = 'data/processed/daily_fact_table.parquet'
FINAL_FORECAST_OUTPUT_PATH = 'data/processed/final_forecast_multi_horizon.parquet'

# [cite_start]Regras de Negócio: Horizontes de previsão (RN05) [cite: 18, 44]
FORECAST_HORIZONS = [7, 14, 30] # dias

def generate_multi_horizon_forecast(df_fact: pd.DataFrame, horizons: List[int]):
    """
    Treina o modelo final e gera previsões para múltiplos horizontes (7, 14, 30 dias).
    (RF05 - Selecionar e salvar previsões P50/P90 para 7, 14 e 30 dias.) [cite_start][cite: 125]
    """
    
    GROUP_COLS = ['normalized_category', 'entities_id']
    df_series_continuous = create_continuous_series(df_fact, GROUP_COLS)
    
    all_forecasts_final = []
    
    groups = df_series_continuous.groupby(GROUP_COLS)
    
    # 1. Iteração por Categoria (Grupo)
    for name, group_df in groups:
        category, entity = name
        
        # Filtro de cold-start (RN04)
        if len(group_df) < 30: 
             continue 

        print(f"Gerando previsões para {category}/{entity} em múltiplos horizontes...")

        try:
            # 2. Treinamento Otimizado para o Horizonte Mais Longo (30 dias)
            # A função optimize_and_forecast retorna apenas 2 valores: (df_forecast_30) e (mape_30).
            df_forecast_30, mape_30, _ = optimize_and_forecast(group_df, horizon=30) 
            
            # [cite_start]3. Geração de Previsões para cada horizonte (RN05) [cite: 18, 44]
            # Filtramos a previsão de 30 dias para cobrir os horizontes 7 e 14.
            
            # Obtemos a data inicial da previsão (próximo dia após o último histórico)
            start_date_forecast = df_forecast_30['date'].min()
            
            for h in horizons:
                
                # Calcula a data final para o horizonte 'h'
                end_date_forecast = start_date_forecast + timedelta(days=h - 1)
                
                # Filtra o DataFrame de 30 dias para o horizonte 'h'
                df_future_forecast = df_forecast_30.loc[
                    (df_forecast_30['date'] >= start_date_forecast) & 
                    (df_forecast_30['date'] <= end_date_forecast)
                ].copy()
                
                # [cite_start]Cria a estrutura de output (RF05) [cite: 125]
                df_future_forecast = df_future_forecast.assign(
                    normalized_category=category,
                    entities_id=entity,
                    horizon=h,
                    # P50_volume e P90_volume já vêm da função optimize_and_forecast
                )
                
                all_forecasts_final.append(df_future_forecast[['date', 'normalized_category', 'entities_id', 'horizon', 'P50_volume', 'P90_volume']])

        except Exception as e:
            # Mantemos o print de erro para depuração
            print(f"Erro ao gerar previsão final para {category}/{entity}: {e}")
    
    if not all_forecasts_final:
        print("Nenhuma previsão gerada. Verifique os filtros de Cold Start.")
        return
        
    # Salvar resultados FINAIS (Tabela de Fato + Previsão)
    df_forecasts = pd.concat(all_forecasts_final)
    
    # [cite_start]🚨 Geração do Dataset completo para o Power BI (RF08) [cite: 125]
    
    # Prepara o Histórico (volume é o real; Pxx é igual ao volume para plotagem histórica)
    df_fact['horizon'] = 0 # 0 indica que é dado histórico
    df_fact['P50_volume'] = df_fact['volume']
    df_fact['P90_volume'] = df_fact['volume']
    
    # 1. Histórico
    df_history_dataset = df_fact.rename(columns={'volume': 'volume_real'}).copy()
    
    # 2. Previsão (volume_real é nulo para o futuro)
    df_forecast_dataset = df_forecasts.assign(volume_real=pd.NA).copy()
    
    # [cite_start]Unifica os datasets (RF08) [cite: 125]
    df_final_dataset = pd.concat([
        df_history_dataset[['date', 'normalized_category', 'entities_id', 'horizon', 'volume_real', 'avg_ttr_hours', 'P50_volume', 'P90_volume']],
        df_forecast_dataset.rename(columns={'P50_volume': 'P50_volume', 'P90_volume': 'P90_volume'})
        [['date', 'normalized_category', 'entities_id', 'horizon', 'volume_real', 'P50_volume', 'P90_volume']]
    ], ignore_index=True)
    
    # [cite_start]Adiciona as colunas de KPI para o dashboard (RF09) [cite: 125]
    df_final_dataset['avg_ttr_hours'] = df_final_dataset['avg_ttr_hours'].fillna(method='ffill')

    # [cite_start]Salva o dataset completo para o Power BI (RF07) [cite: 125]
    powerbi_path = 'data/processed/powerbi_dataset_final.parquet'
    df_final_dataset.to_parquet(powerbi_path, index=False)
    
    print("\n--- MODELO CANDIDATO FINAL PRONTO ---")
    print(f"Dataset Histórico + Previsão (RF08) salvo em: {powerbi_path}")
    print("Fase 2 (Modelagem) concluída. Podemos iniciar a Fase 3 (Implantação/Dashboard).")


if __name__ == "__main__":
    df_fact_table = load_and_prepare_data(FACT_TABLE_PATH)
    generate_multi_horizon_forecast(df_fact_table, FORECAST_HORIZONS)