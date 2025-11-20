import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils import (
    load_and_clean_data, filter_data, create_cohorts, 
    calculate_retention, calculate_rfm, calculate_clv_empirical,
    calculate_clv_formula, calculate_scenario_impact,
    export_segments_for_activation, format_number
)

# Config de la page
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    """Charge les données"""
    return load_and_clean_data('data/processed/online_retail_clean.csv')

def display_filters():
    """Affiche les filtres dans la sidebar"""
    st.sidebar.title("Filtres d'analyse")
    
    # Période
    st.sidebar.subheader("Période")
    min_date = st.session_state.df['InvoiceDate'].min().date()
    max_date = st.session_state.df['InvoiceDate'].max().date()
    
    date_range = st.sidebar.date_input(
        "Sélectionner la période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Pays
    st.sidebar.subheader("Géographie")
    countries = st.session_state.df['Country'].unique().tolist()
    selected_countries = st.sidebar.multiselect(
        "Pays",
        options=countries,
        default=countries[:5] if len(countries) > 5 else countries
    )
    
    # Retours
    st.sidebar.subheader("Retours")
    exclude_returns = st.sidebar.checkbox("Exclure les retours", value=True)
    
    # Montant minimum
    st.sidebar.subheader("Filtres montant")
    min_order = st.sidebar.number_input(
        "Montant minimum de commande",
        min_value=0.0,
        value=0.0,
        step=10.0
    )
    
    return {
        'date_range': date_range if len(date_range) == 2 else (min_date, max_date),
        'countries': selected_countries,
        'exclude_returns': exclude_returns,
        'min_order': min_order
    }

def show_active_filters():
    """Affiche les filtres actifs en haut de chaque page"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Période", f"{st.session_state.filters['date_range'][0]} - {st.session_state.filters['date_range'][1]}", 
                 label_visibility="visible")
    with col2:
        st.metric("Pays", f"{len(st.session_state.filters['countries'])}", 
                 label_visibility="visible")
    with col3:
        if st.session_state.filters['exclude_returns']:
            st.error("🚫 Retours exclus")
        else:
            st.info("✓ Retours inclus")
    with col4:
        st.metric("Seuil commande", f"£{st.session_state.filters['min_order']:.0f}", 
                 label_visibility="visible")
    st.divider()

def page_overview(df_filtered):
    """Page 1: Vue d'ensemble"""
    st.title("Vue d'ensemble - KPIs")
    show_active_filters()
    
    st.divider()
    
    # Calcul des KPIs
    df_no_returns = df_filtered[~df_filtered['IsReturn']]
    
    total_customers = df_no_returns['Customer ID'].nunique()
    total_revenue = df_no_returns['Amount'].sum()
    total_orders = df_no_returns['Invoice'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    # CLV empirique
    clv_empirical, _ = calculate_clv_empirical(df_filtered, months=12)
    
    # Affichage des KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Clients Actifs",
            value=format_number(total_customers, decimals=0),
            help="Nombre total de clients uniques"
        )
    
    with col2:
        st.metric(
            label="Chiffre d'Affaires",
            value=format_number(total_revenue, prefix='£', decimals=1),
            help="Revenu total sur la période"
        )
    
    with col3:
        st.metric(
            label="Panier Moyen",
            value=format_number(avg_order_value, prefix='£', decimals=2),
            help="Valeur moyenne d'une commande"
        )
    
    with col4:
        st.metric(
            label="CLV Moyenne (12 mois)",
            value=format_number(clv_empirical, prefix='£', decimals=0),
            help="Customer Lifetime Value empirique sur 12 mois"
        )
    
    st.divider()
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolution du CA mensuel")
        monthly_revenue = df_no_returns.groupby(df_no_returns['InvoiceDate'].dt.to_period('M'))['Amount'].sum().reset_index()
        monthly_revenue['InvoiceDate'] = monthly_revenue['InvoiceDate'].astype(str)
        
        fig = px.line(monthly_revenue, x='InvoiceDate', y='Amount',
                     labels={'Amount': 'Chiffre d\'Affaires', 'InvoiceDate': 'Mois'})
        fig.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Evolution du nombre de clients")
        monthly_customers = df_no_returns.groupby(df_no_returns['InvoiceDate'].dt.to_period('M'))['Customer ID'].nunique().reset_index()
        monthly_customers['InvoiceDate'] = monthly_customers['InvoiceDate'].astype(str)
        
        fig = px.line(monthly_customers, x='InvoiceDate', y='Customer ID',
                     labels={'Customer ID': 'Nombre de clients', 'InvoiceDate': 'Mois'})
        fig.update_traces(line_color='#2ca02c')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top pays
    st.subheader("Top 10 des pays par CA")
    top_countries = df_no_returns.groupby('Country')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
    
    fig = px.bar(top_countries, x='Amount', y='Country', orientation='h',
                labels={'Amount': 'Chiffre d\'Affaires', 'Country': 'Pays'})
    st.plotly_chart(fig, use_container_width=True)

def page_cohorts(df_filtered):
    """Page 2: Analyse des cohortes"""
    st.title("Analyse des Cohortes")
    show_active_filters()
    
    st.info("Les cohortes regroupent les clients par leur mois de première commande pour suivre leur comportement dans le temps.")
    
    # Créer les cohortes
    df_cohorts = create_cohorts(df_filtered)
    
    # Calculer la rétention
    retention_rate, retention_matrix = calculate_retention(df_cohorts)
    
    # Heatmap
    st.subheader("Matrice de rétention par cohorte")
    st.write("Lecture: Chaque ligne = une cohorte. Chaque colonne = âge en mois. Valeur = % de clients actifs.")
    
    fig = go.Figure(data=go.Heatmap(
        z=retention_rate.values,
        x=[f"M+{i}" for i in retention_rate.columns],
        y=retention_rate.index.astype(str),
        colorscale='RdYlGn',
        text=np.round(retention_rate.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 8},
        colorbar=dict(title="Retention (%)")
    ))
    
    fig.update_layout(
        xaxis_title="Age de la cohorte (mois)",
        yaxis_title="Mois d'acquisition",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenu par âge
    st.subheader("Revenu moyen par age de cohorte")
    
    revenue_by_age = df_cohorts.groupby('CohortAge').agg({
        'Amount': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    revenue_by_age['AvgRevenue'] = revenue_by_age['Amount'] / revenue_by_age['Customer ID']
    
    fig = px.line(revenue_by_age, x='CohortAge', y='AvgRevenue',
                 labels={'CohortAge': 'Age (mois)', 'AvgRevenue': 'Revenu moyen par client'},
                 markers=True)
    fig.update_traces(line_color='#ff7f0e')
    st.plotly_chart(fig, use_container_width=True)
    
    # Focus sur une cohorte spécifique
    st.subheader("Focus sur une cohorte")
    cohort_list = sorted(df_cohorts['CohortMonth'].unique().astype(str))
    selected_cohort = st.selectbox("Sélectionner une cohorte", cohort_list)
    
    if selected_cohort:
        cohort_data = df_cohorts[df_cohorts['CohortMonth'].astype(str) == selected_cohort]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Taille initiale", f"{cohort_data[cohort_data['CohortAge']==0]['Customer ID'].nunique()}")
        with col2:
            total_revenue = cohort_data['Amount'].sum()
            st.metric("CA total généré", f"£{total_revenue:,.0f}")
        with col3:
            avg_age = cohort_data['CohortAge'].max()
            st.metric("Age max observé", f"{avg_age} mois")
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        retention_m1 = retention_rate[1].mean() if 1 in retention_rate.columns else 0
        st.metric("Retention M+1", f"{retention_m1:.1f}%",
                 help="Pourcentage de clients qui reviennent le mois suivant")
    
    with col2:
        retention_m3 = retention_rate[3].mean() if 3 in retention_rate.columns else 0
        st.metric("Retention M+3", f"{retention_m3:.1f}%",
                 help="Pourcentage de clients actifs après 3 mois")
    
    with col3:
        retention_m6 = retention_rate[6].mean() if 6 in retention_rate.columns else 0
        st.metric("Retention M+6", f"{retention_m6:.1f}%",
                 help="Pourcentage de clients actifs après 6 mois")

def page_segments(df_filtered):
    """Page 3: Segmentation RFM"""
    st.title("Segmentation RFM")
    show_active_filters()
    
    st.info("RFM: Recency (récence), Frequency (fréquence), Monetary (valeur). Chaque dimension est scorée de 1 à 5.")
    
    # Calculer RFM
    analysis_date = df_filtered['InvoiceDate'].max()
    rfm_df = calculate_rfm(df_filtered, analysis_date)
    
    # Vue d'ensemble
    st.subheader("Distribution des segments")
    
    segment_summary = rfm_df.groupby('Segment').agg({
        'Customer ID': 'count',
        'Monetary': ['sum', 'mean'],
        'Frequency': 'mean',
        'Recency': 'mean'
    }).reset_index()
    
    segment_summary.columns = ['Segment', 'Nombre', 'CA Total', 'CA Moyen', 'Frequence Moy.', 'Recence Moy.']
    segment_summary = segment_summary.sort_values('CA Total', ascending=False)
    
    # Graphiques
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(segment_summary, x='Segment', y='Nombre',
                    color='CA Total',
                    labels={'Nombre': 'Nombre de clients', 'CA Total': 'CA Total'},
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(segment_summary, values='CA Total', names='Segment',
                    title='Repartition du CA par segment')
        st.plotly_chart(fig, use_container_width=True)
    
    # Table détaillée
    st.subheader("Detail des segments")
    st.dataframe(
        segment_summary.style.format({
            'Nombre': '{:,.0f}',
            'CA Total': '£{:,.0f}',
            'CA Moyen': '£{:,.2f}',
            'Frequence Moy.': '{:.1f}',
            'Recence Moy.': '{:.0f} jours'
        }),
        use_container_width=True
    )
    
    # Segments prioritaires
    st.subheader("Segments prioritaires")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        champions = segment_summary[segment_summary['Segment'] == 'Champions']['Nombre'].values
        st.metric("Champions", format_number(champions[0] if len(champions) > 0 else 0),
                 help="Meilleurs clients")
    
    with col2:
        at_risk = segment_summary[segment_summary['Segment'] == 'A risque']['Nombre'].values
        st.metric("A risque", format_number(at_risk[0] if len(at_risk) > 0 else 0),
                 help="Clients à forte valeur qui n'ont pas acheté récemment")
    
    with col3:
        lost = segment_summary[segment_summary['Segment'] == 'Perdus']['Nombre'].values
        st.metric("Perdus", format_number(lost[0] if len(lost) > 0 else 0),
                 help="Clients inactifs")
    
    # Export
    st.divider()
    st.subheader("Export des segments")
    
    segments_to_export = st.multiselect(
        "Sélectionner les segments à exporter",
        options=rfm_df['Segment'].unique().tolist(),
        default=['Champions', 'Loyaux']
    )
    
    if st.button("Générer CSV"):
        export_df = export_segments_for_activation(rfm_df, segments_to_export)
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Télécharger le CSV",
            data=csv,
            file_name=f"segments_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        st.success(f"{len(export_df)} clients sélectionnés")

def page_scenarios(df_filtered):
    """Page 4: Simulation de scénarios"""
    st.title("Simulation de Scenarios")
    show_active_filters()
    
    st.info("Simulez l'impact de différentes actions marketing sur la CLV et la retention.")
    
    # Calcul baseline
    df_no_returns = df_filtered[~df_filtered['IsReturn']]
    
    total_customers = df_no_returns['Customer ID'].nunique()
    total_revenue = df_no_returns['Amount'].sum()
    total_orders = df_no_returns['Invoice'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    customer_orders = df_no_returns.groupby('Customer ID')['Invoice'].nunique()
    avg_frequency = customer_orders.mean()
    
    # Calculer la retention
    df_cohorts = create_cohorts(df_filtered)
    retention_rate, _ = calculate_retention(df_cohorts)
    avg_retention = retention_rate[1].mean() / 100 if 1 in retention_rate.columns else 0.3
    
    baseline_metrics = {
        'customers': total_customers,
        'total_revenue': total_revenue,
        'avg_order_value': avg_order_value,
        'purchase_frequency': avg_frequency,
        'retention': avg_retention,
        'margin': 0.30,
    }
    
    # CLV baseline
    baseline_metrics['clv'] = calculate_clv_formula(
        avg_order_value,
        avg_frequency,
        avg_retention,
        discount_rate=0.1,
        periods=12
    )
    
    # Paramètres du scénario
    st.subheader("Parametres du scenario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        retention_change = st.slider(
            "Variation de retention (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5
        )
        
        margin_change = st.slider(
            "Variation de marge (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5
        )
    
    with col2:
        discount = st.slider(
            "Remise moyenne (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=5
        )
        
        discount_rate = st.slider(
            "Taux d'actualisation (%)",
            min_value=0,
            max_value=30,
            value=10,
            step=5
        ) / 100
    
    # Calculer le scénario
    scenario_params = {
        'retention_change': retention_change,
        'margin_change': margin_change,
        'discount': discount,
        'discount_rate': discount_rate
    }
    
    scenario_metrics = calculate_scenario_impact(baseline_metrics, scenario_params)
    
    # Résultats
    st.divider()
    st.subheader("Impact du scenario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clv_delta = ((scenario_metrics['clv'] - baseline_metrics['clv']) / baseline_metrics['clv'] * 100)
        st.metric(
            "CLV Moyenne",
            format_number(scenario_metrics['clv'], prefix='£', decimals=0),
            delta=f"{clv_delta:+.1f}%"
        )
    
    with col2:
        retention_delta = (scenario_metrics['retention'] - baseline_metrics['retention']) * 100
        st.metric(
            "Taux de retention",
            f"{scenario_metrics['retention']*100:.1f}%",
            delta=f"{retention_delta:+.1f}pp"
        )
    
    with col3:
        aov_delta = ((scenario_metrics['avg_order_value'] - baseline_metrics['avg_order_value']) / baseline_metrics['avg_order_value'] * 100)
        st.metric(
            "Panier Moyen",
            format_number(scenario_metrics['avg_order_value'], prefix='£', decimals=2),
            delta=f"{aov_delta:+.1f}%"
        )
    
    # Graphique de comparaison
    st.subheader("Comparaison Baseline vs Scenario")
    
    comparison_df = pd.DataFrame({
        'Metrique': ['CLV', 'CA Total', 'Retention (%)'],
        'Baseline': [
            baseline_metrics['clv'],
            baseline_metrics['total_revenue'],
            baseline_metrics['retention'] * 100
        ],
        'Scenario': [
            scenario_metrics['clv'],
            scenario_metrics.get('total_revenue', baseline_metrics['total_revenue']),
            scenario_metrics['retention'] * 100
        ]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Baseline', x=comparison_df['Metrique'], y=comparison_df['Baseline'],
                        marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Scenario', x=comparison_df['Metrique'], y=comparison_df['Scenario'],
                        marker_color='darkblue'))
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)

def page_export(df_filtered):
    """Page 5: Export"""
    st.title("Export et Plan d'Action")
    show_active_filters()
    
    st.info("Exportez les données et visualisations pour partager vos analyses.")
    
    # RFM pour export
    rfm_df = calculate_rfm(df_filtered)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export des données")
        
        # Export CSV données filtrées
        csv_filtered = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger données filtrées (CSV)",
            data=csv_filtered,
            file_name=f"data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Export CSV RFM
        csv_rfm = rfm_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger analyse RFM (CSV)",
            data=csv_rfm,
            file_name=f"rfm_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.info("💡 Pour exporter les graphiques en PNG, faites clic droit sur chaque graphique > 'Download plot as png'")
    
    with col2:
        st.subheader("Résumé de la session")
        
        st.write(f"Période: {st.session_state.filters['date_range'][0]} - {st.session_state.filters['date_range'][1]}")
        st.write(f"Transactions: {len(df_filtered):,}")
        st.write(f"Clients uniques: {df_filtered['Customer ID'].nunique():,}")
        st.write(f"Pays: {len(st.session_state.filters['countries'])}")
        st.write(f"Retours: {'Exclus' if st.session_state.filters['exclude_returns'] else 'Inclus'}")
    
    st.divider()
    
    st.subheader("Recommandations")
    st.markdown("""
    **Actions prioritaires:**
    
    - Champions: Programme de parrainage, accès VIP
    - A risque: Campagne de réactivation avec offre personnalisée
    - Nouveaux: Séquence d'onboarding
    - Perdus: Évaluer le ROI d'une récupération
    
    **Optimisations testées:**
    - Amélioration de 5% de la retention → impact sur CLV
    - Calibrer les remises selon le segment RFM
    - Focus sur cohortes à forte valeur
    """)

def main():
    """Application principale"""
    
    # Charger les données
    if 'df' not in st.session_state:
        try:
            st.session_state.df = load_data()
        except:
            st.error("Impossible de charger les données. Vérifiez le chemin du fichier.")
            st.stop()
    
    # Filtres
    st.session_state.filters = display_filters()
    
    # Appliquer les filtres
    df_filtered = filter_data(
        st.session_state.df,
        date_range=st.session_state.filters['date_range'],
        countries=st.session_state.filters['countries'],
        exclude_returns=st.session_state.filters['exclude_returns'],
        min_order=st.session_state.filters['min_order']
    )
    
    # Navigation
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Cohortes", "Segments RFM", "Scenarios", "Export"]
    )
    
    # Afficher la page
    if page == "Overview":
        page_overview(df_filtered)
    elif page == "Cohortes":
        page_cohorts(df_filtered)
    elif page == "Segments RFM":
        page_segments(df_filtered)
    elif page == "Scenarios":
        page_scenarios(df_filtered)
    elif page == "Export":
        page_export(df_filtered)
    
    # Footer
    st.sidebar.divider()
    st.sidebar.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

if __name__ == "__main__":
    main()