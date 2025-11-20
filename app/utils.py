import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_clean_data(filepath):
    """Charge et nettoie les données"""
    df = pd.read_csv(filepath)
    
    # Nettoyage des colonnes
    df.columns = df.columns.str.strip()
    
    # Conversion de la date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Enlever les lignes sans Customer ID
    df = df.dropna(subset=['Customer ID'])
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # Identifier les retours
    df['IsReturn'] = df['Invoice'].astype(str).str.startswith('C')
    
    # Calculer le montant
    df['Amount'] = df['Quantity'] * df['Price']
    
    return df

def filter_data(df, date_range=None, countries=None, exclude_returns=False, min_order=0):
    """Applique les filtres"""
    df_filtered = df.copy()
    
    if date_range:
        # Convertir les dates en datetime pour la comparaison
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df_filtered = df_filtered[
            (df_filtered['InvoiceDate'] >= start_date) & 
            (df_filtered['InvoiceDate'] <= end_date)
        ]
    
    if countries and len(countries) > 0:
        df_filtered = df_filtered[df_filtered['Country'].isin(countries)]
    
    if exclude_returns:
        df_filtered = df_filtered[~df_filtered['IsReturn']]
    
    if min_order > 0:
        df_filtered = df_filtered[df_filtered['Amount'] >= min_order]
    
    return df_filtered

def create_cohorts(df):
    """Crée les cohortes mensuelles"""
    df = df[~df['IsReturn']].copy()
    
    # Date de première commande par client
    df['CohortMonth'] = df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')
    df['OrderMonth'] = df['InvoiceDate'].dt.to_period('M')
    
    # Calculer l'âge en mois
    df['CohortAge'] = (df['OrderMonth'] - df['CohortMonth']).apply(lambda x: x.n)
    
    return df

def calculate_retention(df):
    """Calcule la matrice de rétention"""
    # Grouper par cohorte et âge
    cohort_data = df.groupby(['CohortMonth', 'CohortAge'])['Customer ID'].nunique().reset_index()
    cohort_data.columns = ['CohortMonth', 'CohortAge', 'Customers']
    
    # Taille de chaque cohorte au départ
    cohort_sizes = cohort_data[cohort_data['CohortAge'] == 0].set_index('CohortMonth')['Customers']
    
    # Créer le pivot
    retention_matrix = cohort_data.pivot_table(
        index='CohortMonth',
        columns='CohortAge',
        values='Customers'
    )
    
    # Calculer les taux de rétention
    retention_rate = retention_matrix.divide(cohort_sizes, axis=0) * 100
    
    return retention_rate, retention_matrix

def calculate_rfm(df, analysis_date=None):
    """Calcule les scores RFM"""
    if analysis_date is None:
        analysis_date = df['InvoiceDate'].max()
    
    df_rfm = df[~df['IsReturn']].copy()
    
    # Calculer RFM par client
    rfm = df_rfm.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,
        'Invoice': 'nunique',
        'Amount': 'sum'
    }).reset_index()
    
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    # Scores de 1 à 5
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Score combiné
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Créer les segments
    rfm['Segment'] = rfm['RFM_Score'].apply(segment_rfm)
    
    return rfm

def segment_rfm(score):
    """Attribue un segment selon le score RFM"""
    r, f, m = int(score[0]), int(score[1]), int(score[2])
    
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyaux'
    elif r >= 4 and f <= 2:
        return 'Nouveaux'
    elif r >= 3 and f <= 2:
        return 'Prometteurs'
    elif r <= 2 and f >= 4:
        return 'A risque'
    elif r <= 2 and f >= 2:
        return 'Hibernation'
    else:
        return 'Perdus'

def calculate_clv_empirical(df, months=12):
    """CLV empirique basée sur les cohortes"""
    df_cohorts = create_cohorts(df[~df['IsReturn']])
    
    # Revenu moyen par client par âge
    clv_by_age = df_cohorts.groupby('CohortAge').agg({
        'Amount': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    
    clv_by_age['AvgRevenue'] = clv_by_age['Amount'] / clv_by_age['Customer ID']
    
    # Somme sur N mois
    clv = clv_by_age[clv_by_age['CohortAge'] < months]['AvgRevenue'].sum()
    
    return clv, clv_by_age

def calculate_clv_formula(avg_order_value, purchase_frequency, retention_rate, discount_rate=0.1, periods=12):
    """CLV avec formule fermée"""
    if retention_rate >= (1 + discount_rate):
        return avg_order_value * purchase_frequency * periods
    
    margin_per_period = avg_order_value * purchase_frequency
    
    # CLV actualisée sur N périodes
    clv = sum([
        (margin_per_period * (retention_rate ** t)) / ((1 + discount_rate) ** t)
        for t in range(periods)
    ])
    
    return clv

def calculate_scenario_impact(baseline_metrics, scenario_params):
    """Calcule l'impact d'un scénario"""
    scenario_metrics = baseline_metrics.copy()
    
    # Appliquer les changements
    if 'retention_change' in scenario_params:
        scenario_metrics['retention'] *= (1 + scenario_params['retention_change'] / 100)
        scenario_metrics['retention'] = min(scenario_metrics['retention'], 1.0)
    
    if 'margin_change' in scenario_params:
        scenario_metrics['margin'] *= (1 + scenario_params['margin_change'] / 100)
    
    if 'discount' in scenario_params:
        scenario_metrics['avg_order_value'] *= (1 - scenario_params['discount'] / 100)
    
    # Recalculer la CLV
    scenario_metrics['clv'] = calculate_clv_formula(
        scenario_metrics['avg_order_value'],
        scenario_metrics['purchase_frequency'],
        scenario_metrics['retention'],
        scenario_params.get('discount_rate', 0.1)
    )
    
    scenario_metrics['total_revenue'] = scenario_metrics['clv'] * scenario_metrics['customers']
    
    return scenario_metrics

def export_segments_for_activation(rfm_df, segments_to_export):
    """Prépare un CSV avec les segments sélectionnés"""
    export_df = rfm_df[rfm_df['Segment'].isin(segments_to_export)].copy()
    export_df = export_df[['Customer ID', 'Segment', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']]
    
    return export_df

def format_number(value, prefix='', suffix='', decimals=0):
    """Formate les nombres pour l'affichage"""
    if pd.isna(value):
        return 'N/A'
    
    if abs(value) >= 1_000_000:
        return f"{prefix}{value/1_000_000:.{decimals}f}M{suffix}"
    elif abs(value) >= 1_000:
        return f"{prefix}{value/1_000:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{value:.{decimals}f}{suffix}"