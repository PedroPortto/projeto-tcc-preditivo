import pandas as pd
import numpy as np

FILE_PATH = 'data/processed/daily_fact_table.parquet'

def inspect_fact_table(file_path: str):
    """Carrega a tabela fato e exibe estatísticas para validação."""
    try:
        # Tenta carregar o arquivo Parquet
        df_fact = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo {file_path} não encontrado. Certifique-se de que foi gerado.")
        return

    print("-" * 50)
    print(f"VALIDAÇÃO DA TABELA FATO: {file_path}")
    print("-" * 50)
    
    # 1. Inspeção Geral e Tipos de Dados
    print("1. Tipos de Dados e Preenchimento (info()):")
    df_fact.info()
    
    # 2. Amostra do Conteúdo (Head)
    print("\n2. Amostra das primeiras linhas:")
    print(df_fact.head())

    # 3. Validação do Mapeamento de Categorias (RN01)
    # Verifica quantas categorias únicas foram geradas. Espera-se 36 + 'OUTROS'
    num_unique_categories = df_fact['normalized_category'].nunique()
    outros_count = df_fact[df_fact['normalized_category'] == 'OUTROS']['volume'].sum()

    print("\n3. Validação de Categorias (RN01):")
    print(f"Total de Categorias Únicas Mapeadas: {num_unique_categories}")
    print(f"Volume de tickets agrupados em 'OUTROS': {outros_count} (Correto se for baixo)")
    print("Top 5 Categorias por Volume (Sanidade):")
    print(df_fact.groupby('normalized_category')['volume'].sum().nlargest(5))

    # 4. Validação de Features de Tempo (RF02)
    print("\n4. Validação de Features de Tempo (RF02):")
    print(f"Média de is_weekend: {df_fact['is_weekend'].mean():.4f} (Deve ser ~0.28)")
    print(f"Dias no dataset: {df_fact['date'].nunique()}")
    print(f"Data Inicial: {df_fact['date'].min().strftime('%Y-%m-%d')}")
    print(f"Data Final: {df_fact['date'].max().strftime('%Y-%m-%d')}")
    
    # 5. Validação do KPI TTR (KPI02)
    print("\n5. Estatísticas do TTR Médio (KPI02):")
    print(df_fact['avg_ttr_hours'].describe())
    
    # Confirmação da granularidade diária
    print(f"\nTotal de linhas no Fato Diário: {len(df_fact)}")
    print("-" * 50)

if __name__ == "__main__":
    inspect_fact_table(FILE_PATH)