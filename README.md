# Projet : Customer Analytics Dashboard (Cohortes, RFM, CLV)

## Structure
- notebooks → exploration & nettoyage
- app → Streamlit
- data/raw → données brutes
- data/processed → données nettoyées

## Lancer l'app :
pip install -r requirements.txt
streamlit run app/app.py

## Description
Cette application permet à l’équipe marketing de :
- Diagnostiquer la rétention par cohortes d’acquisition
- Calculer la CLV globale et par segment (empirique et formule fermée)
- Segmenter les clients via RFM et prioriser les actions CRM
- Simuler des scénarios marketing (marge, remise, rétention) avec impact immédiat sur CLV, CA et rétention
- Exporter les données filtrées et segments activables en CSV

## Filtres disponibles
- Période d’analyse (glissante)
- Pays
- Seuil de commande
- Inclure / exclure les retours

## Pages principales
1. **Overview** : KPIs globaux (clients, CA, panier moyen, CLV)
2. **Cohortes** : heatmap de rétention, revenu moyen par âge, focus sur une cohorte
3. **Segments RFM** : tableau des segments, priorités d’activation, export CSV
4. **Scenarios** : sliders pour marge, rétention, remise, taux d’actualisation; comparaison baseline vs scénario
5. **Export** : export des données filtrées et RFM + recommandations

