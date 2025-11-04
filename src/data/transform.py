import pandas as pd
from datetime import date
from typing import List, Dict
import numpy as np

# Regras de Negócio: Feriados (RN01) - AJUSTE ESTA LISTA PARA O SEU PERÍODO DE DADOS
# A lista abaixo é um EXEMPLO para 2025 no Brasil.
FERIADOS_BR: List[date] = [
    date(2025, 1, 1), date(2025, 2, 25), date(2025, 2, 26), date(2025, 4, 18),
    date(2025, 4, 21), date(2025, 5, 1), date(2025, 9, 7), date(2025, 10, 12),
    date(2025, 11, 2), date(2025, 11, 15), date(2025, 11, 20), date(2025, 12, 25)
]

# Dicionário de Mapeamento de Categorias (RN01)
# TODAS AS CHAVES ESTÃO EM MINÚSCULAS E SEM ESPAÇOS EXTRAS (STRIPPED)
CATEGORY_MAPPING = {
    # ------------------------------------
    # ACESSO
    # ------------------------------------
    "acesso / acesses": "ACESSO_GERAL",
    "acesso / acesses > acesso vpn / vpn acess": "ACESSO_VPN",
    "acesso / acesses > criar usuario / create user": "ACESSO_CRIACAO_USUARIO",
    "acesso / acesses > permissao de pasta / folder permission": "ACESSO_PASTA_PERMISSAO",
    "acesso / acesses > resetar senha / reset password": "ACESSO_RESET_SENHA",
    
    # ------------------------------------
    # E-MAIL
    # ------------------------------------
    "e-mail": "EMAIL_GERAL",
    "e-mail > alterar dados do email / change email information": "EMAIL_ALTERACAO_DADOS",
    "e-mail > criar conta de e-mail / create email account": "EMAIL_CRIACAO_CONTA",
    "e-mail > problemas com e-mail / email problems": "EMAIL_PROBLEMAS",
    "e-mail > resetar senha de e-mail / reset email password": "EMAIL_RESET_SENHA",
    
    # ------------------------------------
    # HARDWARE / EQUIPAMENTOS
    # ------------------------------------
    "hardware": "HARDWARE_GERAL",
    "hardware > computador travando / computer freezing": "HARDWARE_COMPUTADOR_TRAVANDO",
    "hardware > configuração de computador / computer configuration": "HARDWARE_CONFIG_COMPUTADOR",
    "hardware > monitor com problema / monitor issue": "HARDWARE_MONITOR",
    "hardware > notebook não liga / laptop won't turn on": "HARDWARE_NOTEBOOK_NAOLIGA",
    "hardware > trocar teclado/mouse / replace keyboard/mouse": "HARDWARE_TROCAR_TECLADO_MOUSE",

    # ------------------------------------
    # IMPRESSORAS
    # ------------------------------------
    "impressoras / printers": "IMPRESSORA_GERAL",
    "impressoras / printers > duvidas / doubts": "IMPRESSORA_DUVIDAS",
    "impressoras / printers > instalar impressora / install printer": "IMPRESSORA_INSTALACAO",
    "impressoras / printers > não imprime / not printing": "IMPRESSORA_NAO_IMPRIME",
    "impressoras / printers > outros problemas / other issues": "IMPRESSORA_OUTROS_PROBLEMAS",

    # ------------------------------------
    # LARK (Nova Categoria Mapeada)
    # ------------------------------------
    "lark": "LARK_GERAL",
    "lark > alteração de fluxo / flow change": "LARK_ALTERACAO_FLUXO",
    "lark > criação de fluxo / flow creation": "LARK_CRIACAO_FLUXO",

    # ------------------------------------
    # REDE / NETWORK (Nova Categoria Mapeada)
    # ------------------------------------
    "rede / network": "REDE_GERAL",
    "rede / network > sem internet / no internet": "REDE_SEM_INTERNET",
    "rede / network > wi-fi instável / unstable wi-fi": "REDE_WIFI_INSTAVEL",

    # ------------------------------------
    # SOFTWARE / SISTEMAS
    # ------------------------------------
    "software": "SOFTWARE_GERAL",
    "software > atualizar sistema/programa / update system/program": "SOFTWARE_ATUALIZACAO",
    "software > erro em software / software error": "SOFTWARE_ERRO",
    "software > instalar programa / install program": "SOFTWARE_INSTALACAO",
    "software > licença / license": "SOFTWARE_LICENCA",

    # ------------------------------------
    # SUPORTE GERAL
    # ------------------------------------
    "suporte geral / general support": "SUPORTE_GERAL",
    "suporte geral / general support > outros problemas / other issues": "SUPORTE_OUTROS_PROBLEMAS",
    "suporte geral / general support > solicitação sem categoria / uncategorized request": "SUPORTE_SEM_CATEGORIA",
    "suporte geral / general support > tirar duvidas / ask questions": "SUPORTE_DUVIDAS",
}

def map_categories(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Aplica uma taxonomia unificada de categorias (RN01) de forma tolerante."""
    
    # CORREÇÃO CRÍTICA: 
    # 1. Converte a coluna para string e minúsculas (.lower())
    # 2. Remove espaços extras no início e fim (.strip())
    df['category_path_lower_stripped'] = df['category_path'].astype(str).str.lower().str.strip().fillna('')
    
    # Mapeia caminhos (agora limpos) para nomes padronizados (maiúsculas)
    # Se não houver match, cai em 'OUTROS'
    df['normalized_category'] = df['category_path_lower_stripped'].apply(
        lambda x: mapping.get(x, 'OUTROS')
    )
    
    # Remove a coluna temporária
    df = df.drop(columns=['category_path_lower_stripped'])
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de calendário (RF02) e trata dados de tempo/TTR (KPI02)."""
    
    # 1. Tratamento de Datas
    df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')
    df['date'] = df['opened_at'].dt.normalize()
    
    df = df.dropna(subset=['date']) 
    df = df.drop_duplicates(subset=['id'], keep='first') # Remove duplicatas (RN03)
    
    # 2. Tratamento de Tempo de Resolução (TTR - KPI02)
    
    # CORREÇÃO: Converter 'time_to_resolve' para numérico antes de comparar
    df['time_to_resolve'] = pd.to_numeric(df['time_to_resolve'], errors='coerce') 

    # Tenta usar time_to_resolve (segundos). Se inválido ou zero, usa hours_to_solve.
    df['ttr_hours'] = np.where(
        # Verifica se não é nulo (notna) E se é maior que zero
        df['time_to_resolve'].notna() & (df['time_to_resolve'] > 0),
        df['time_to_resolve'] / 3600, # Converte segundos para horas
        df['hours_to_solve']         # Usa a coluna calculada (TIMESTAMPDIFF) como fallback
    )

    # Trata NaNs no TTR com a mediana (RN03)
    ttr_median = df['ttr_hours'].median()
    df['ttr_hours'] = df['ttr_hours'].fillna(ttr_median)
    
    # 3. Features de Calendário (RF02)
    df['day_of_week'] = df['date'].dt.dayofweek # 0=Segunda, 6=Domingo
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_holiday'] = df['date'].apply(lambda x: x.date() in FERIADOS_BR).astype(int)
    
    return df

def create_daily_fact_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o DataFrame de tickets para criar a Tabela Fato Diária (RF02).
    """
    
    # Agregação para volume (Alvo do modelo) e média de TTR (KPI02) por Categoria/Entidade
    daily_fact = df.groupby(['date', 'normalized_category', 'entities_id']).agg(
        volume=('id', 'count'),
        avg_ttr_hours=('ttr_hours', 'mean'), 
    ).reset_index()
    
    # Mesclar Features de Calendário
    calendar_features = df[['date', 'day_of_week', 'is_weekend', 'month', 'year', 'is_holiday']].drop_duplicates()
    daily_fact = pd.merge(daily_fact, calendar_features, on='date', how='left')

    daily_fact = daily_fact.sort_values(by=['date', 'normalized_category', 'entities_id']).reset_index(drop=True)
    
    return daily_fact

def process_data(df_tickets: pd.DataFrame, category_mapping: Dict[str, str]) -> pd.DataFrame:
    """Função principal para processar e gerar a tabela fato."""
    
    df_features = feature_engineering(df_tickets)
    df_mapped = map_categories(df_features, category_mapping)
    df_fact = create_daily_fact_table(df_mapped)

    print(f"Tabela Fato Diária gerada. Total de linhas (dias/categorias): {len(df_fact)}")
    return df_fact

if __name__ == "__main__":
    from src.data.extract import get_glpi_tickets # Requer a importação do extract
    # Carrega dados brutos (deve ter sido gerado pelo extract.py)
    input_path = 'data/raw/glpi_tickets_raw.csv'
    try:
        df_raw = pd.read_csv(input_path, encoding='utf-8') 
    except FileNotFoundError:
        print(f"ERRO: Arquivo {input_path} não encontrado. Execute 'python -m src.data.extract' primeiro.")
        exit()
    except UnicodeDecodeError:
         print("Aviso: Tentando recarregar com codificação 'latin-1'...")
         df_raw = pd.read_csv(input_path, encoding='latin-1')

    print(f"Iniciando transformação com {len(df_raw)} tickets brutos...")
    df_fact = process_data(df_raw, CATEGORY_MAPPING)
    
    # Armazenamento da Tabela Fato Processada (RF06, RF07)
    output_path = 'data/processed/daily_fact_table.parquet'
    df_fact.to_parquet(output_path, index=False)
    print(f"\nETL Concluído. Tabela Fato Diária salva em: {output_path}")