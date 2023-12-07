#!/usr/bin/env python


# standard imports
from contextlib import contextmanager
import time
import pickle
import json
import logging
import sys

# third party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import shap

# custom
import lightgbm_with_simple_features as lgbmsf

#------------------ Logging --------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

# console handler
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
    )

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# ----------------------------------------------------------------

with open('config.json', 'r') as f:
    config = json.load(f)
NUM_ROWS = config['NUM_ROWS']
FP_READ = config['READ']
FP_WRITE = config['WRITE']

with open('model_params.json', 'r') as f:
    model_params = json.load(f)
    
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    logger.info("{} - done in {:.0f}s".format(title, time.time() - t0))


def drop_cols(df:pd.DataFrame,
              threshold:float=.8,
              sample_frac:float=.1
              ):
    """
    Remove a feature among each pair of features if their correlation exceeds the threshold. Because of the size of the
    dataset (large number of features and rows), the computation is performed only on a subset with sample_frac rows
    from the original dataset. 

    Parameters
    ----------
    df: pandas.DataFrame
        The dataset
    threshold: float, default=.8
        the threshold to use to decide whether two features are "too" correlated.
    sample_frac: float, default=.1
        The fraction of the dataset to use to calculate the correlations among features. Must be between 0 and 1.

    Returns
    -------
    df: pandas.DataFrame
        The dataset with redundant features removed. 
    """

    # calculer les corrélations sur tout le jeu de données prend trop de temps
    sample_df = df.sample(frac=sample_frac)
    
    with timer("Computing features correlation matrix"):
        corr = sample_df.drop('SK_ID_CURR', axis=1).corr().abs()
        
    # triangle supérieur des corrélations
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype('bool'))

    # Sélection des colonnes au-dessus du seuil de corrélation
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    cleaned_df = df.drop(columns=to_drop)

    logger.info(f'{len(to_drop)} colonnes redondantes ont été écartées du jeu de données')

    return cleaned_df


def prepare_for_dashboard(ids:List=None):
    """
    Prepare the data to use with the dashboard app. 

    Currently the data is loaded when the dashboard is launched. In order to keep things light,
    only a fraction of the original dataset is used, and this fraction corresponds to the data selected
    during the call to the create_datasets function (through the num_rows argument).

    Parameters
    ----------
    ids : List
        List of indices
    """

    # Création des tables à partir des clients sélectionnés pour affichage dans le dashboard
    logger.info('Processing sample files...')

    for i in [
    'POS_CASH_balance.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'application_train.csv',
    'bureau.csv',
    'previous_application.csv',
    ]:
        with timer("Process " + i):
            df = pd.read_csv(FP_READ + i)
            df = df[df.SK_ID_CURR.isin(ids)]
            logger.info(f'{i.split(".")[0]} df shape: {df.shape}')
            df.to_csv(FP_WRITE + i, index_label=False)

    skidbureau = pd.read_csv(FP_READ + 'bureau.csv').SK_ID_BUREAU
    bb = pd.read_csv(FP_READ + 'bureau_balance.csv')
    bb = bb[bb.SK_ID_BUREAU.isin(skidbureau)]
    logger.info(f'bureau_balance df shape: {bb.shape}')
    file_name = 'bureau_balance.csv'
    bb.to_csv(FP_WRITE + file_name, index_label=False)
    logger.info("Done")

    return


def create_datasets(load_from_existing:bool=True,
                    num_rows:int=NUM_ROWS,
                    sample_fraction:float=.25,
                    threshold:float=.8,
                    save:bool=True
                    ):
    """
    Import the dataset or recreate the dataset from the tables. Clean it and save it.
     
    If load_from_existing is True, import the dataset using FP_READ. Otherwise, recreate the
    dataset. If recreating the dataset, choose the number of rows to keep with the num_rows argument. Remove redundant features by
    calculating the correlations and choosing only one feature if correlation exceeds the threshold value. If save is True, save the dataset
    using the FP_READ path.
    
    Parameters
    ----------
    load_from_existing: bool, default True
        Load the dataset if True, recreate it if False. Raise an error if True and there is no dataset to be found.
    num_rows: int, default NUM_ROWS
        Number of rows to keep if recreating the dataset. Default is the value stored in the config file as NUM_ROWS.
    sample_fraction: float, default .25
        The fraction of the dataset to use to compute the correlations between features.
    threshold: float, default .8
        The threshold to use to remove features based on their correlation.
    save: bool, default True
        Whether to save the created dataset or not.

    Returns
    -------
    cleaned_df: pandas.DataFrame
        The imported/recreated and cleaned dataset.

    """
    
    if load_from_existing:
        logger.info('Load from existing')
        try:
            cleaned_df = pd.read_csv(FP_READ + 'data.csv')
        except Exception as e:
            logger.info("No dataset found! Try setting load_from_existing to False, or check the filepath is correct.")
            raise e

    else:
        logger.info('Recreate dataset')
        # création du dataset complet
        df = lgbmsf.join_df(num_rows=num_rows)

        # On retire une feature pour chaque paire de features corrélées à plus de 80%
        cleaned_df = drop_cols(df, threshold=threshold, sample_frac=sample_fraction)
        cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if save:
            # enregistre sample_data pour utilisation dans le dashboard 
            with timer('Saving cleaned dataset to {}...'.format(FP_READ + 'data.csv')):
                cleaned_df.to_csv(FP_READ + 'data.csv', index_label=False)
    
    # vérification de l'indexation du jeu de données
    assert 'SK_ID_CURR' in cleaned_df.columns
    for col in ['SK_ID_BUREAU','SK_ID_PREV','index']:
        assert col not in cleaned_df.columns

    return cleaned_df


def prepare_data(df:pd.DataFrame, test_size:float=.05):
    """
    Prepare the training and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to be split into training and test sets.
    test_size : float, default .05
        The size of the test dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test : pandas.DataFrame or pandas.Series
        The train/test sets.
    
    """

    if df is None:
        fp = FP_READ + 'data.csv'
        try:
            with timer('Loading data from {}'.format(fp)):
                    df = pd.read_csv(fp)
                
        except:
            logger.info("You must provide a path or a dataset")
    
    selected = [f for f in df.columns if f != 'TARGET']
    X = df.filter(items=selected)
    y = df.TARGET

    # Séparation en jeux d'entraînement/test. On ne garde que 10% du jeu total pour le jeu de test
    # en vue du déploiement via heroku qui impose des limites sur la taille des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    logger.info("Train data shape: {}".format(X_train.shape))
    logger.info("Test data shape: {}".format(X_test.shape))

    # Sauvegarde du jeu de test
    X_test.index = pd.Index(range(X_test.shape[0]))
    with timer('Saving test data at {}'.format(FP_WRITE + 'features_test.csv')):
        X_test.to_csv(FP_WRITE + 'features_test.csv', index_label=False)
    
    return X_train, X_test, y_train, y_test


def run_model(X_train, X_test, y_train, y_test):
    """Réalise l'entraînement du modèle (LightGBM)."""
    
    X_train = X_train.drop('SK_ID_CURR', axis=1).to_numpy()
    X_test = X_test.drop('SK_ID_CURR', axis=1).to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()


    best_params = model_params['best_params']
    model = LGBMClassifier(
        early_stopping_round=50,
        objective='binary',
        metric='AUC',
        silent=False,
        verbosity=-1,
        **best_params
        )  

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 50)
    
    logger.info("Entraînement avec les paramètres suivants: ")
    for k, v in best_params.items():
        logger.info("{} = {}".format(k, v))
    
    logger.info("Résultats de l'entraînement")
    logger.info("---------------------------")
    
    # Performance sur le jeu d'entraînement
    logger.info("Performance sur le jeu d'entraînement : {:.3f}".format(model.score(X_train, y_train)))
    
    # Performance en généralisation du meilleur modèle sur le jeu de test
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    logger.info("Performance en généralisation sur le jeu de test : {:.3f}".format(roc_auc_score(y_test, y_pred_proba)))
    logger.info("Espérance de la variable y (jeu de test): {:.2f}".format(y_pred.mean()))
    
    logger.info("Espérance de la probabilité de faire défaut (jeu de test): {}".format(y_pred_proba.mean(0).round(3)))
    
    logger.info("Classification Report")
    logger.info(f'''Classification report:
                ----------------------
    {classification_report(y_test, model.predict(X_test))}
    ''')

    # Sauvegarde du modèle
    fp = FP_WRITE + 'lgb.pkl'
    logger.info(f'Saving the model at {fp}')
    with open(fp, 'wb') as f:
        pickle.dump(model, f)

    return model
    

# Display/plot feature importance
def plot_feature_importances(df, threshold = 0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    fig, ax = plt.subplots(figsize = (12, 6))
    fig.subplots_adjust(left=.4, right=.99)
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.draw()
    
    # Cumulative importance plot
    plt.figure(figsize = (8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.draw()
    
    return df


def create_shap_data(model=None, \
                     data=None, 
                     background_data=None,
                     #path_to_save=config['SAVE_TO'], 
                     #FP_READ=config['READ_FROM'], 
                     sample_size=10000,
                     features_to_keep=None
                     ):
    """Calculate SHAP values for the model"""

    if data is None:
        data = pd.read_csv(FP_WRITE + 'features_test.csv').drop(columns='SK_ID_CURR')
    data.drop(columns='SK_ID_CURR', errors='ignore', inplace=True)

    tree_explainer = shap.TreeExplainer(model,
                                        data=background_data,
                                        feature_perturbation='interventional',
                                        model_output='probability'
                                        )
    
    base_value = tree_explainer.expected_value
    mean_default = np.mean(model.predict_proba(background_data)[:, 1])
    logger.info("SHAP base value (espérance de la probabilité de faire défaut): {:.3f}".format(base_value))
    logger.info("espérance de la probabilité de faire défaut sur le jeu d'entraînement: {:.3f}".format(mean_default))

    shap_values = tree_explainer(data)
    shap_values.values = shap_values.values.astype('float32')
    
    # test on first client
    total_shap = base_value + shap_values.values[0].sum()
    predicted_proba = model.predict_proba(data.iloc[[0]])[0][1]
    logger.info("Total SHAP values (reconstructed probability of default) for client #0: {:.3f}".format(total_shap))
    logger.info("Expected probability of default (from model.predict_proba) for client #0: {:.3f}".format(predicted_proba))
    assert round(total_shap, 3) == round(predicted_proba, 3)

    with open(FP_WRITE + 'shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values.values, f)
    
    with open(FP_WRITE + 'base_value.pkl', 'wb') as f:
        pickle.dump(base_value, f)

    return

def main(num_rows=10000,
         sample_fraction=.1,
         test_size=.05,
         threshold_feat_importance=.99,
         sample_size=2000
):
    logger.info('Creating datasets')
    data = create_datasets(
        load_from_existing=True,
        num_rows=num_rows, 
        sample_fraction=sample_fraction
        )
    
    logger.info('Preparing data')
    X_train, X_test, y_train, y_test = prepare_data(data, test_size=test_size)

    # Récupération des clients présents dans le jeu de test
    logger.info('Select data for the dashboard')
    ids = X_test.SK_ID_CURR.unique()
    prepare_for_dashboard(ids, config=config)
    
    logger.info('Training model...')
    model = run_model(X_train, X_test, y_train, y_test)
    
    logger.info('Computing feature importances')
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = [c for c in X_train.columns if c not in ['SK_ID_CURR']]
    feature_importance_df["importance"] = model.feature_importances_
    norm_feature_importances = plot_feature_importances(feature_importance_df, threshold=threshold_feat_importance)

    # Extraction des features les plus importantes
    condition = 'cumulative_importance < @threshold_feat_importance'
    features_to_keep = norm_feature_importances.query(condition)['feature'].to_list()
    assert 'SK_ID_CURR' not in features_to_keep
    
    # Sauvegarde des features
    model_params['most_important_features'] = features_to_keep
    
    # mise à jour des paramètres du modèle
    with open("model_params.json", "w") as f:
        json.dump(model_params, f)

    logger.info('Calculating shap values')
    bg_data = (X_train
               .drop(columns='SK_ID_CURR')
               .sample(sample_size)
               )
    
    create_shap_data(
        model=model,
        data=X_test,
        background_data=bg_data,  
        features_to_keep=features_to_keep
        )
    plt.show()

if __name__ == '__main__':
    main(num_rows=None, sample_fraction=.25, test_size=.05, sample_size=5000)