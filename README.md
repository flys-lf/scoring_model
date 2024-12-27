# scoring_model
Projet réalisé dans le cadre d'un parcours certifiant de Data Scientist OpenClassroom : Implémentez un modèle de scoring (2 mois / 80h).

# Contexte :
Une société financière, nommée "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
Elle souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

# Mission :
Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.

Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.

Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

# Guideline
- Installer les dépendances :
```
poetry install
poetry shell
```
- Lancer MLFlow : 
```
mlflow server --host 127.0.0.1 --port 8080
```

- Repositoty contenant le code de déploiement de l'API et du dashboard Streamlit ici : https://github.com/flys-lf/deploiement_api