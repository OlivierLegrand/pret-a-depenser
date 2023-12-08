# Prêt à dépenser

Il s'agit du 7ème projet de la certification Data Scientist dispensée par OpenClassrooms (sur les huit projets au total). Dans ce projet on cherche à mettre en place un algorithme de scoring visant à prédire la probabilité qu'un client fasse faillite, puis à présenter les résultats et leur interprétation sous la forme d'un dashboard interactif.  


# I Présentation du projet

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser",  qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

### Votre mission
1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
2. Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

Michaël, votre manager, vous incite à sélectionner un kernel Kaggle pour vous faciliter la préparation des données nécessaires à l’élaboration du modèle de scoring. Vous analyserez ce kernel et l’adapterez pour vous assurer qu’il répond aux besoins de votre mission.

Vous pourrez ainsi vous focaliser sur l’élaboration du modèle, son optimisation et sa compréhension.

### Spécifications du dashboard
Michaël vous a fourni des spécifications pour le dashboard interactif. Celui-ci devra contenir au minimum les fonctionnalités suivantes :

* Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
* Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
* Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.


# II Comment utiliser ce répertoire
## II.1 Initialisation et contenu du répertoire

Le répertoire doit en principe contenir les fichiers suivants:
- Le notebook contenant l'analyse exploratoire, l'entraînement du modèle et les essais d'interprétation du modèle
- les fichiers lightgbm_with_simple_features.py et create_model.py renfermant les scripts nécessaires au fonctionnement du modèle et des applications
- les dossiers prediction-api et dashboard-app contenant les fichiers nécessaires au déploiement des applications en local (le déploiement sur le web n'est plus disponible)

La première étape consiste à lancer l'execution du script create_model.py:

	python create_model.py 

L'execution du script va créer le dossier sample_data et y placer les fichiers csv des données, le modèle entraîné et enregistré au format pickle (lgb.pkl), ainsi que les fichiers contenant les données utiles pour l'interprétation du modèle (shap_values.pkl et base_value.pkl). Il crée également le dossier project_files dans lequel les données téléchargées et le jeu de données créer pour l'entraînement du modèle sont placés. 

Après execution du script create_model.py, le repertoire doit donc ressembler à ça:
```
.
└── root  /
    ├── .git
    ├── .gitignore
    ├── HomeCredit_columns_description.csv
    ├── LightGBMClassifier_final.ipynb
    ├── README.md
    ├── config.json
    ├── create_model.py
    ├── lightgbm_with_simple_features.py
    ├── model_params.json
    ├── requirements.txt
    ├── data/
    │   ├── project_files/
    │   │   ├── data.csv
    │   │   └── ...
    │   └── sample_data/
    │       ├── base_value.pkl
    │       ├── features_test.csv
    │       ├── lgb.pkl
    │       ├── shap_values.pkl
    │       └── ...
    ├── dashboard-app/
    │   ├── Procfile
    │   ├── config.json
    │   ├── home_credit.py
    │   ├── requirements.txt
    │   └── runtime.txt
    └── prediction-api/
        ├── Model.py
        ├── Procfile
        ├── app.py
        ├── config.json
        ├── requirements.txt
        └── runtime.txt
```

Le code est largement inspiré du kernel kaggle disponible à cette adresse: https://www.kaggle.com/c/home-credit-default-risk/data\
L'algorithme de prédiction utilisé est LightGBM https://lightgbm.readthedocs.io/

## II.2 Procédures pour lancer les applications en local

L'application peut maintenant être lancée:

	cd prediction-api
	python app.py

Ceci lancera l'API de prédiction. Il est nécessaire de lancer l'API de prédiction d'abord car elle est ensuite appelée par l'application dashboard. Pour lancer l'application dashboard, se replacer à la racine, puis:

	cd dashboard-app
	python home_credit.py


# III Prédiction de la capacité de remboursement d'emprunt des clients de Home Credit
Home Credit est une institution financière internationale non bancaire fondée en 1997 en République tchèque et basée aux Pays-Bas. La société opère dans 9 pays et se concentre sur les prêts à tempérament principalement aux personnes ayant peu ou pas d'antécédents de crédit. [Wikipedia](https://en.wikipedia.org/wiki/Home_Credit). A cause de la nature même de sa clientèle, Home Credit a recours à des sources d'informations variées - notamment des informations liées à l'utilisation des services de téléphonie et aux transactions effectuées par les clients - pour prédire la capacité de remboursement des clients. C'est une partie de ces données (anonymisées) que Home Credit a mis en ligne  (https://www.kaggle.com/c/home-credit-default-risk/data), et sur lesquelles le présent travail repose.

## III.1 Présentation du modèle
Les données sont constituées de huit tables, liées les unes aux autres via une ou plusieurs clés comme indiqué sur le shéma ci-dessous [home_credit](home_credit.png)
Les descriptions des différentes tables peuvent être consultées [ici](https://www.kaggle.com/c/home-credit-default-risk/data).\
Le modèle utilise toutes ces tables, Les jointures et le _feature engineering_ étant réalisés par la fonction main() du module p7.py, très étroitement inspirée du kernel kaggle [LightGBM with simple features](https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/script).

Le problème à résoudre concerne **la probabilité qu'un client ne rembourse pas son crédit**. La cible qu'on cherche à prédire est une variable **binaire**: 0 signifie qu'un client rembourse son crédit (no default), 1 que le client ne rembourse pas son crédit (default). Il s'agit donc d'un problème de classification, et on peut relever certaines caractéristiques importantes à prpos du jeu de données:
- un grand nombre de features (plusieurs centaines)
- les features sont de type mixte: continu et catégoriel
- il s'agit d'un jeu de données **déséquilibré**, dans le sens ou la proportion de clients n'ayant pas remboursé leur crédit est d'environ **9%** (et non 50% comme attendu dans le cas d'un jeu de données équilibré). Cela va fortement peser dans les stratégies d'optimisation du modèle.

## III.2 Entraînement du modèle

### III.2.1 Algorithme de prédiction
L'algorithme utilisé est [LightGBM](https://lightgbm.readthedocs.io). Il s'agit d'un algorithme de la famille des arbres de décision boostés et qui possède des caractéristiques le rendant particulièrement efficace, comme la méthode de croissance des arbres ou encore le traitement des variables catégorielles ([voir la doc officielle](https://lightgbm.readthedocs.io/en/latest/Features.html)), en plus des qualités intrinsèques de cette famille d'algorithmes, qui permet, en combinant un grand nombre d'apprenants faibles permet de réaliser un modèle particulièrement performant. 
Les contreparties de cette efficacité sont multiples:
- le fort risque de surapprentissage
- le grand nombre d'hyperparamètres ajustables et dont une détermination optimale nécessite une recherche sur grille dans un espace des paramètres de grande taille. Ainsi, on choisit pour la détermination des meilleurs hyperparamètres l'algorithme RandomSearchCV qui va permettre de parcourir une portion significative de l'espace des paramètres en n'utilisant qu'un nombre restreint d'itérations, permettant ainsi un gain de temps sensible par rapport à la méthode standard GridSearchCV au prix toutefois d'une évaluation plus ou moins suboptimale en comparaison.\

### III.2.2 Optimisation des hyperparamètres
Pour effectuer l'optimisation des hyperparamètres, le jeu de données est scindé une première fois en une paire entraînement/test dans les proportions 0.7/0.3. Ceci est réalisé grâce à la fonction train_test_split de la bibliothèqye scikit-learn en prenant en compte le déséquilibre des classes grâce au paramètre stratify=True, ce qui permet d'avoir des jeux d'entraînement et de test qui reproduisent aussi fidèlement que possible les distributions relatives des deux classes dans le jeu de données global. Chaque modèle est évalué au travers d'une validation croisée sur 5 folds stratifiés (sklearn.model_selection.StratifiedKFolds) afin de prendre en compte le déséquilibre des classes. Les hyperparamètres ajustés sont les suivants:
- le nombre maximal de feuilles (num_leaves) - permet de prévenir le surapprentissage
- le taux d'apprentissage (learning_rate) - permet de trouver un compromis entre efficacité et performance "brute"
- le nombre de boosts (n_estimators) - pour augmenter la peformance
- le nombre de features retenues pour faire croître chaque arbre (colsample_bytree), en fraction du nombre total de features
- le nombre d'individus minimal requis pour faire "pousser" une feuille - prévention du surapprentissage
- le poids affecté aux individus de la classe minoritaire (scale_pos_weight): ce paramètre est pris en compte par la fonction de coût lors de l'entraînement du modèle, en affectant une pénalité plus forte aux éléments de la classe minoritaire incorrectement classifiés
- bagging_fraction, qui consiste à ne construire les arbres que sur un échantillon d'individus tirés au sort (avec remplacement): permet de réduire le risque de surapprentissage

### III.2.3 Fonction de coût et métrique d'évaluation
LightGBM peut-être utilisé pour réaliser des régressions aussi bien que des classifications. Dans le cas qui nous concerne, la fonction de coût est la fonction *log-loss*, ou encore *negative log likelihood* et le *coût* associé à une prédiction s'écrit:

	L(y, p) = -(ylog(p) + (1-y)log(1-p))

où y représente la "vraie" valeur de la cible, et p la probabilité associée à la classe 1. Le coût associé à la totalité du jeu de données est simplement donné par la somme des coûts des échantillons individuels, et les paramètres du modèle sont alors ajustés de manière à minimiser ce nombre. Une fois le modèle entraîné, son évaluation se fait au moyen de la métrique choisie, la plus courante étant la précision (accuracy) dans le cas des problèmes de classification. 

L'une des faiblesse de cette méthode, dans notre cas, vient du caractère déséquilibré du jeu de données: en effet, dans la mesure où la classe minoritaire représente environ 10% des échantillons, on peut obtenir une précision d'environ **90%** simplement en prédisant toujours la classe majoritaire. Fait d'autant plus gênant que la prédiction correcte de la classe minoritaire est cruciale dans notre cas, plus importante que la prédiction correcte de la classe majoritaire - il vaut mieux, pour un organisme de crédit, refuser de prêter à un client qui aurait remboursé (faux positif) que d'accepter de prêter à un client non solvable (faux négatif). Pour éviter ces écueils, on agit sur deux leviers:
- la fonction de coût
- les métriques utilisées pour évaluer la qualité du modèle
 
**la fonction de coût**\
Comme évoqué plus haut, LightGBM possède un paramètre *scale_pos_weight* qui permet de multiplier les echantillons de la classe minoritaire par un poids de manière à pénaliser plus fortement les échantillons de la classe minoritaire incorrectement classifiés. En effet, la fonction de coût s'écrit alors, dans le cas où y=1:
	
	y = 1: L(y, p) = -wlog(p)
 	
où w représente le poids. La prédiction incorrecte est donc associée à un coût w fois plus grand que dans le cas où l'échantillon appartient à la classe majoritaire

	y = 0: L(y, p) = -log(1-p)

Le coût total étant égal à la somme sur tous les échantillons, ceci permet bien d'obtenir une fonction de coût qui pénalisera *in fine* autant les mauvaises classifications associées à chacune des classes si w est ajusté de telle sorte que:
	
	w = nb éch. classe majoritaire/nb éch. classe minoritaire

**la métrique d'évaluation**\
De même, il faut choisir une métrique qui permette de mieux rendre compte de la qualité de la classification pour chacune des classes. Plusieurs outils peuvent répondre à cela:
- Area Under Receiver Operating Characteristic (AUROC): la ROC est une courbe construite en reportant, pour chaque valeur de seuil de discrimination, le taux de vrais positifs (fraction des positifs qui sont effectivement détectés, précision) en fonction du taux de faux positifs (fraction des négatifs qui sont incorrectement détectés, antispécificité). Cette courbe part du point (0, 0) où tout est classifié comme "négatif", et arrive au point (1, 1) où tout est classifié "positif". Un classifieur aléatoire tracera une droite allant de (0, 0) à (1, 1), donnant une AUROC égale à 0.5, et tout classifieur "meilleur" sera représenté par une courbe située au-dessus de cette droite et possèdera ainsi une AUROC supérieure à 0.5. Nous utilisons principalement cette métrique pour évaluer le modèle, conformément au souhait de l'établissement Home Credit.
- la matrice de confusion, qui permet de visualiser directement le taux de vrais positifs (sensibilité, ou *recall*), ainsi que le ratio du nombre de vrais positifs sur la somme des vrais positifs et des faux positifs (précision).

### III.2.4 Sélection des features pour simplifier et optimiser le modèle
Une sélection des features les plus importantes peut-être réalisée, à la fois dans un but de simplification et d'optimisation du modèle:
- simplification: on peut ainsi réduire le nombre de features, et surtout écarter celles dont la valeur prédictive est proche de 0
- optimisation: en ne gardant que les features les plus importantes on réduit le risque de surapprentissage.

### III.2.5 Résumé de la méthodologie d'entraînement
En réssumé, les étapes mises en oeuvre sont les suivantes: 
1. Chargement du jeu de données complet.
2. Un première sélection des features en ne gardant qu'une feature pour chaque paire de features corrélées à plus de 80%. Cette étape réduit le nombre de features de 797 à un peu plus de 530.
3. Séparation entraînement/test une fois pour la recherche sur grille des meilleurs paramètres, puis sélection des features les plus pertinentes via la méthode de feature_importances_ de la classe LGBMClassifier. Cette sélection doit se faire sur un modèle aussi bon que possible, ce qui rend nécessaire la recheche sur grille de bons paramètres déjà à cette étape. Cette étape permet d'obtenir une réduction de 530+ à environ 234 features.
4. Chargement du jeu de données complet, avec restriction aux features sélectionnées auparavant.
5. Deuxième recherche sur grille, puis entraînement sur le jeu d'entraînement complet avec les meilleurs paramètres.

# IV Interprétation du modèle
On cherche, dans un but de compréhension du modèle mais aussi de communication, à interpréter le modèle c'est à dire à déterminer:
* les variables les plus importantes dans notre jeu de données, c'est-à-dire celles qui ont le plus d'impact sur le score global obtenu par l'estimateur - ici F1 et/ou AUROC. Il s'agit alors de l'interprétation globale.
* Pour chaque prédiction, quelles sont les variables qui ont principalement conduit l'estimateur à faire la prédiction en question. Il s'agit alors de l'interprétation locale.

LightGBM représente une sorte de "boîte noire": il s'agit d'un algorithme *non-paramétrique*, et contrairement au cas de la régréssion linéaire nous n'avons pas à notre disposition des coefficients (évalués lors de l'entraînement) associés à chaque variable et qui permettent directement d'interpréter le poids affecté à chaque variable par le modèle. On choisit donc, pour l'interprétation globale, d'utiliser le calcul des *feature importances* via la méthode feature_importances_ de la classe LGBMClassifier. Cette méthode consiste à classer les features selon le nombre de fois qu'elles sont utilisées par le modèle, c'est-à-dire le nombre d'embranchements de l'arbre de décision auxquels elles sont associées. Plus le nombre d'embranchements est important, plus la feature est importante.
Dans notre cas, cette méthode est utilisée deux fois: une première fois pour sélectionner les variables les plus  imoportantes, une deuxième fois pour réaliser l'interprétation globale.

Pour l'interprétation locale du mmodèle, on fait appel à la bibliothèque SHAP, et au calcul des *shap values*. Il s'agit ici d'une méthode directement adaptée de la théorie des jeux, et qui consiste à évaluer l'influence de chaque variable sur la prédiction faite par le modèle (ici la probabilité de non-remboursement). Le point de départ de cette méthode est l'évaluation d'un score de base, qui correspond en fait à l'espérance de la variable cible, et qui peut s'interpréter comme la probabilité moyenne sur l'ensemble du jeu de données qu'un individu fasse défaut. Dans notre cas cette probabilité est de 40%. L'estimateur, parce qu'il a été entraîné, renvoie une prédiction sensiblement différente et le calcul des *shap values* permet d'associer à chaque variable sa part de la différence entre la probabilité calculée par l'estimateur et l'espérance. Les détails de l'implémentation de cette bibliothèque peuvent être consultés [ici](https://shap-lrjball.readthedocs.io/en/latest/).

# V Seuil de classification
Le modèle entraîné affiche un score de rappel - i.e. un taux de vrais positifs - de 71%, et une précision de 97% pour la classe majoritaire. Ceci a pour conséquence que l'établissement doit s'attendre à:
- classer incorrectement des clients à risque de faire défaut comme des clients sûrs
- classer incorrectement des clients "sûrs" comme étant "à risque".

Ceci a pour conséquence de réduire les gains attendus par rapport à ce qu'ils seraient si l'établissement pouvait classer parfaitement chaque client: d'une part parce qu'un certain nombre de clients se verront accorder un crédit alors même que leur capacité réelle de remboursement ne permettra pas de rembourser le crédit, d'autre part parce qu'un certain nombre de clients jugés "à risque" sont en faits "sûrs" privant ainsi l'établissement d'une source de revenus supplémentaire. Pour optimiser les gains espérés, on peut cependant ajuster le seuil de classification.

Le seuil de classification est le seuil de probabilité au-delà duquel le modèle classe un individu dans la catégorie 1. Par défaut ce seuil est égal à 0,5: un individu dont la probabilité de faire défaut est supérieure à 50% sera classé positivement. Le modèle de prédiction étant imparfait, rien n'assure cependant que ce seuil soit optimal. Dans notre cas, en supposant que l'établissement prête 100 et attende un rendement de 5 sur chaque crédit alloué, et en supposant par ailleurs une perte de 100 par client faisant défaut, on peut montrer que le seuil de classification optimal, c'est-à-dire celui maximisant les gains est égal à 35%, tandis qu'au-delà d'un seuil à 64% l'établissement doit s'attendre à des pertes nettes. Pour arriver à ce résultat, on considère sur la base de 100 clients:
- un gain de 5\*Vrais Négatifs
- une perte de 100\*Faux négatifs
- un manque à gagner de 5\*Faux positifs

les nombres de Vrais négatifs, faux négatifs et faux positifs étant indiqués par la matrice de confusion calculée pour différentes valeurs de seuils.

# V Limites et améliorations possibles
Même si l'estimateur entraîné réalise une meilleure classification que le classifieur aléatoire (AUROC de 0.784 vs. 0.5 pour le classifieur aléatoire), le modèle souffre quand même d'une performance limitée, avec notamment une précision de 0.18 sur la classe minoritaire obtenue sur le jeu de test. On peut imaginer les pistes suivantes pour améliorer les résultats: 

* Entraînement de l'estimateur:
	- la capacité du modèle à gérer directement les variables catégorielles n'est pas exploitée, mais il serait intéressant de voir si cela conduit à de meilleures performances.
	- bien qu'une recherche sur grille ait été effectuée, il se peut que les valeurs retenues correspondent à un extremum local. Une recherche plus exhaustive pourrait peut-être conduire à de meilleurs résultats
	- Déterminer de nouvelles variables via des considérations métiers








