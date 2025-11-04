import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from typing import List, Dict, Tuple
import warnings
from datetime import timedelta

# Ignorar warnings do Prophet e Pandas para clareza
warnings.filterwarnings('ignore')

# Definição de Caminhos
FACT_TABLE_PATH = 'data/processed/daily_fact_table.parquet'
METRICS_OUTPUT_PATH = 'data/processed/model_metrics.csv'
FORECAST_OUTPUT_PATH = 'data/processed/final_forecast.parquet'

# Regras de Negócio: Horizontes de previsão (RN05)
FORECAST_HORIZONS = [7, 14, 30] # dias

def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Carrega o dataframe fato e garante a coluna de data/alvo."""
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_continuous_series(df: pd.DataFrame, group_by_cols: List[str]) -> pd.DataFrame:
    """
    Cria uma série temporal contínua para cada grupo (preenchendo dias faltantes com volume=0).
    A Modelagem exige séries contínuas (RF02).
    """
    
    # 1. Encontra o range completo de datas
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # 2. Cria um MultiIndex com todas as combinações (Data x Categoria x Entidade)
    unique_groups = df[group_by_cols].drop_duplicates()
    
    # Cria um DataFrame base com todas as combinações de data e grupo
    df_base = pd.MultiIndex.from_product(
        [date_range] + [unique_groups[col].unique() for col in unique_groups.columns],
        names=['date'] + group_by_cols
    ).to_frame(index=False)
    
    # 3. Merge com os dados existentes e preenche volume NaN com 0
    df_series = pd.merge(df_base, df, on=['date'] + group_by_cols, how='left')
    df_series['volume'] = df_series['volume'].fillna(0)
    
    # Re-cria features de calendário para os novos dias preenchidos (se necessário)
    df_series['day_of_week'] = df_series['date'].dt.dayofweek
    df_series['is_weekend'] = df_series['day_of_week'].isin([5, 6]).astype(int)
    df_series['month'] = df_series['date'].dt.month
    df_series['year'] = df_series['date'].dt.year
    
    # Feriados: Assume-se que 'is_holiday' do original é 0 para os dias de fill (simplificação)
    df_series['is_holiday'] = df_series['is_holiday'].fillna(0).astype(int)

    return df_series.sort_values(by=group_by_cols + ['date'])

def run_prophet_model(df_series: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, float]:
    """
    Treina o modelo Prophet, calcula MAPE e gera previsões P50/P90.
    """
    # Prophet requer colunas 'ds' (data) e 'y' (valor alvo)
    df_prophet = df_series[['date', 'volume']].rename(columns={'date': 'ds', 'volume': 'y'})
    
    # 1. Backtesting Simples: Usa os últimos 30 dias para validação (como baseline)
    VALIDATION_DAYS = 30
    
    if len(df_prophet) <= VALIDATION_DAYS:
        return pd.DataFrame(), 1.0 # Retorna 100% de erro se não for possível treinar/validar

    train_size = len(df_prophet) - VALIDATION_DAYS
    df_train = df_prophet.iloc[:train_size]
    df_test = df_prophet.iloc[train_size:]
    
    # 2. Treinamento para MAPE (Modelo 1)
    m_mape = Prophet(
        weekly_seasonality=True, 
        seasonality_mode='multiplicative'
    )
    m_mape.add_country_holidays(country_name='BR')
    
    m_mape.fit(df_train)
    
    # 3. Previsão para teste (para cálculo de MAPE)
    future_test = df_test.drop(columns=['y'])
    forecast_test = m_mape.predict(future_test)
    
    # Cálculo do MAPE (Métrica de Sucesso CA-S1)
    mape = mean_absolute_percentage_error(df_test['y'].values, forecast_test['yhat'].values)
    
    # 4. Previsão Final (h=30 dias) - USANDO UM NOVO OBJETO PROPHET
    m_final = Prophet(
        weekly_seasonality=True, 
        seasonality_mode='multiplicative'
    )
    m_final.add_country_holidays(country_name='BR')

    # Treina o NOVO OBJETO com TODOS os dados históricos
    m_final.fit(df_prophet)
    
    # Cria o dataframe futuro (horizonte de 30 dias)
    future = m_final.make_future_dataframe(periods=horizon)
    forecast_final = m_final.predict(future)
    
    # Selecionar P50 (yhat) e P90 (yhat_upper) - RF05
    forecast_output = forecast_final[['ds', 'yhat', 'yhat_upper']].tail(horizon)
    forecast_output.columns = ['date', 'P50_volume', 'P90_volume']
    
    return forecast_output, mape

def train_and_forecast_all(df_fact: pd.DataFrame):
    """Itera sobre todos os grupos (categorias/entidades), treina e salva previsões/métricas."""
    
    GROUP_COLS = ['normalized_category', 'entities_id']
    
    df_series_continuous = create_continuous_series(df_fact, GROUP_COLS)
    
    all_metrics = []
    all_forecasts = []
    
    groups = df_series_continuous.groupby(GROUP_COLS)
    
    for name, group_df in groups:
        category, entity = name
        
        # Filtro de cold-start (RN04)
        if group_df['volume'].sum() < 10: # Mudando o filtro para um valor menor, pois a agregação é feita por grupo/dia
             print(f"Pulando grupo {category}/{entity}: Volume total < 10 (Cold Start).")
             continue 

        # Filtra categorias com menos de 30 dias de dados para não falhar o backtest inicial
        if len(group_df) < 30:
            print(f"Pulando grupo {category}/{entity}: Histórico insuficiente (< 30 dias).")
            continue
            
        print(f"Treinando Prophet (Baseline) para {category}/{entity}...")

        try:
            # Treinar para o horizonte mais longo (30 dias)
            forecast_30, mape_30 = run_prophet_model(group_df, horizon=30)
            
            # Adicionar metadados à previsão
            forecast_30['normalized_category'] = category
            forecast_30['entities_id'] = entity
            forecast_30['horizon'] = 30
            all_forecasts.append(forecast_30)
            
            # Salvar métricas (MAPE inicial do backtest de 30 dias)
            all_metrics.append({
                'normalized_category': category,
                'entities_id': entity,
                'model': 'Prophet_Baseline',
                'horizon': 30,
                'MAPE': mape_30 * 100, # Convertendo para percentual
                'success_status': 'PASS' if mape_30 * 100 <= 20 else 'FAIL' # Meta por categoria (MAPE <= 20%)
            })

        except Exception as e:
            print(f"Erro ao modelar {category}/{entity}: {e}")
    
    # Salvar resultados
    df_metrics = pd.DataFrame(all_metrics)
    df_forecasts = pd.concat(all_forecasts)
    
    df_metrics.to_csv(METRICS_OUTPUT_PATH, index=False)
    df_forecasts.to_parquet(FORECAST_OUTPUT_PATH, index=False)
    
    print("\n--- RESULTADOS GLOBAIS (BASELINE) ---")
    print(f"Total de Categorias Modeladas: {df_metrics['normalized_category'].nunique()}")
    print(f"MAPE Global Médio: {df_metrics['MAPE'].mean():.2f}%")
    print(f"Métricas salvas em: {METRICS_OUTPUT_PATH}")
    print(f"Previsões salvas em: {FORECAST_OUTPUT_PATH}")
    
    # Checagem final da meta global
    if df_metrics['MAPE'].mean() <= 15: # CA-S1
        print("\n✅ Sucesso: MAPE Global <= 15% (Critério CA-S1 - BASELINE ATINGIU!).")
    else:
        print("\n⚠️ O MAPE Global ainda está acima da meta de 15%. Precisamos avançar para XGBoost/LightGBM (Etapa 6).")

if __name__ == "__main__":
    df_fact_table = load_and_prepare_data(FACT_TABLE_PATH)
    train_and_forecast_all(df_fact_table)