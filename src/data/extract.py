from datetime import datetime, timedelta
import pandas as pd
from src.utils.database import fetch_data

def get_glpi_tickets(months_history: int = 12) -> tuple[pd.DataFrame, str]:
    """
    Extrai a base histórica de tickets do GLPI com as colunas essenciais.
    """
    end_date = datetime.now()
    # Pega o primeiro dia do mês de 'months_history' meses atrás
    start_date = end_date - timedelta(days=30 * months_history) 
    
    # A consulta usa os nomes de coluna confirmados (t.id, t.date, c.completename, etc.)
    query_base = f"""
    SELECT
        t.id,
        t.date AS opened_at,
        t.solvedate,
        t.closedate,
        t.status,
        t.priority,
        t.itilcategories_id,
        c.completename AS category_path,
        t.entities_id,
        t.time_to_resolve, -- Campo nativo do GLPI para TTR (em segundos ou outro formato dependendo da config)
        -- Cálculo de horas de resolução para segurança, caso time_to_resolve não seja ideal:
        TIMESTAMPDIFF(HOUR, t.date, t.solvedate) AS hours_to_solve 
    FROM glpi_tickets t
    LEFT JOIN glpi_itilcategories c ON c.id = t.itilcategories_id
    WHERE t.date >= '{start_date.strftime('%Y-%m-%d')}'
      AND t.date < '{end_date.strftime('%Y-%m-%d')}'
    ORDER BY t.date;
    """
    
    df = fetch_data(query_base)
    return df, query_base

if __name__ == "__main__":
    # Teste de extração com 12 meses de histórico
    df_tickets, sql_query = get_glpi_tickets(months_history=12)
    
    # Salva o dataframe bruto para referência
    output_path = 'data/raw/glpi_tickets_raw.csv'
    if not df_tickets.empty:
        df_tickets.to_csv(output_path, index=False)
        print(f"\nExtração concluída. Dados brutos salvos em: {output_path}")