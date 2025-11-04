import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

load_dotenv()

def get_db_engine():
    """Cria e retorna o objeto engine de conexão SQLAlchemy para MYSQL"""
    #Usamos o conector 'mysql+pymysql' para MySQL
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")

    #Formato da url de conexão para SQLAlchemy
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

    try:
        engine = create_engine(DATABASE_URL)
        print("Conexão com o banco de dados estabelecida com sucesso.")
        return engine
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None
    
def fetch_data(query: str) -> pd.DataFrame:
    """Executa uma consulta SQL e retorna o resultado em um DataFrame do Pandas."""
    engine = get_db_engine()
    if engine is None:
        return pd.DataFrame() # Retorna DataFrame vazio em caso de falha

    try:
        # Usa o Pandas para ler a consulta diretamente
        df = pd.read_sql(text(query), engine.connect())
        print(f"Dados extraídos. Total de linhas: {len(df)}")
        return df
    except Exception as e:
        print(f"Erro ao executar a consulta: {e}")
        return pd.DataFrame()
