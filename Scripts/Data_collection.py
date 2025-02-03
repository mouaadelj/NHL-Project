#from comet_ml import Experiment
import re
import os
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import os
import json
from tqdm import tqdm

def load_all_seasons(base_path):
    all_data = {}
    csv_path = 'all_seasons_data.csv'

    if os.path.exists(csv_path):
        print("Chargement des données depuis le fichier CSV existant.")
        return pd.read_csv(csv_path)
    
    seasons = [s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))]

    for season in tqdm(seasons, desc='Seasons', position=0):
        season_path = os.path.join(base_path, season)
        json_files = [f for f in os.listdir(season_path) if f.endswith('.json')]

        for json_file in tqdm(json_files, desc=f'Loading {season}', position=1, leave=False):
            json_path = os.path.join(season_path, json_file)
            with open(json_path, 'r') as f:
                game_data = json.load(f)
                
            game_id = json_file.split('.')[0]
            all_data[game_id] = game_data
    
    df = pd.DataFrame.from_dict(all_data)
    return df

def distance_goal(x: float, y: float):
    """
    Calculer la distance entre le tir et le filet

    Parameters
    ----------
    x: float
        Coordonnée en x du tir
    y: float
        Coordonnée en y du tir
    
    Returns
    -------
    distance: array
        Distance entre le tir et le filet pour tous les tirs
    """
    # Les coordonnées du filet ont été définies comme le centre de la ligne des buts.
    # Sachant que la ligne des buts est située à 11 pieds de la bande et que la patinoire
    # a une largeur de 200 pieds, nous pouvons calculer la coordonnées en x en faisant
    # 200/2 - 11 = 89. La coordonnée en y est simplement 0.
    x_goal, y_goal = 89, 0

    # Calculer la distance entre le tir et le filet
    distance = np.sqrt((x_goal - np.abs(x))**2 + (y_goal - np.abs(y))**2)

    return distance


def angle_goal(x: float, y: float):
    """
    Calculer l'angle entre le tir et le filet

    Parameters
    ----------
    x: float
        Coordonnée en x du tir
    y: float
        Coordonnée en y du tir

    Returns
    -------
    angle: array
        Angle entre le tir et le filet pour tous les tirs
    """
    # Les coordonnées du filet ont été définies comme le centre de la ligne des buts.
    # Sachant que la ligne des buts est située à 11 pieds de la bande et que la patinoire
    # a une largeur de 200 pieds, nous pouvons calculer la coordonnées en x en faisant
    # 200/2 - 11 = 89. La coordonnée en y est simplement 0.
    x_goal, y_goal = 89, 0
    epsilon = 1e-10  # A tiny value to prevent division by zero
    # Calculer l'angle entre le tir et le filet
    angle = np.arctan((y_goal - y) / (x_goal - np.abs(x) + epsilon))
    # Convertir l'angle en degrés
    angle = np.rad2deg(angle)

    return angle

def is_goal(data: pd.DataFrame):
    """
    Encoder la variable booléenne `goalFlag` en variable binaire

    Parameters
    ----------
    data: DataFrame
        Données des tirs
    
    Returns
    -------
    is_goal: array
        Variable binaire indiquant si le tir est un but ou non
    """
    is_goal = LabelEncoder().fit_transform(data['goalFlag'])
    
    return is_goal

def empty_goal(data: pd.DataFrame):
    """
    Encoder la variable booléenne `noGoalie` en variable binaire

    Parameters
    ----------
    data: DataFrame
        Données des tirs

    Returns
    -------
    empty_goal: array
        Variable binaire indiquant si le filet était désert ou non
    """
    empty_goal = LabelEncoder().fit_transform(data['noGoalie'])

    return empty_goal

def create_features1(data: pd.DataFrame, pattern: str, outname: str):
    """
    Fonction pour créer les nouvelles caractéristiques à partir des données spécifiées
    tel que demandé dans la section Ingénierie des caractéristiques I du Milestone 2 et 
    les sauvegarder dans un fichier csv.

    Parameters
    ----------
    data: DataFrame
        Données nettoyées provenant du Milestone 1
    pattern: str
        Regex pour sélectionner certaines données. Si None, toutes les données dans data
        seront utilisées.
    outname: str
        Nom du fichier csv dans lequel les nouvelles caractéristiques seront sauvegardées
    """
    # Instancier un nouveau DataFrame
    new_data = pd.DataFrame()

    # Isoler les données des saisons régulières de 2016-2017 à 2019-2020
    pattern = re.compile(pattern)
    if pattern is not None:
        data = data[data['gameId'].astype(str).str.match(pattern)]
        data.reset_index(inplace=True)


    # Créer la variable distance_goal
    new_data['shot_distance'] = distance_goal(data['coord_x'], data['coord_y'])
    # Créer la variable angle_goal
    new_data['shot_angle'] = angle_goal(data['coord_x'], data['coord_y'])
    # Créer la variable is_goal
    new_data['is_goal'] = is_goal(data)
    # Créer la variable empty_goal
    #new_data['empty_goal'] = empty_goal(data)
    
    #new_data['empty_goal'].fillna(0, inplace=True)
    #new_data.dropna(inplace=True)
    new_data['is_goal'].fillna(0, inplace=True)
    new_data['shot_angle'].fillna(0, inplace=True)
    new_data['shot_distance'].fillna(0, inplace=True)

    new_data.to_csv(f'../data/derivatives/{outname}', index=False)
    
    

def create_features2(data: pd.DataFrame, data_milestone_1: pd.DataFrame, pattern: str):
    """
    Fonction pour ajouter de nouvelles caractéristiques aux données existantes et
    sauvegarder le résultat dans le même fichier CSV.

    Parameters
    ----------
    data: DataFrame
        Données nettoyées provenant du Milestone 1.
    pattern: str
        Regex pour sélectionner certaines données. Si None, toutes les données dans data
        seront utilisées.
    """
    # Isoler les données des saisons régulières de 2016-2017 à 2019-2020
    pattern = re.compile(pattern)
    if pattern is not None:
        data = data[data['gameId'].astype(str).str.match(pattern)]
        data.reset_index(inplace=True)

    data = data.copy()

    # Ajout des nouvelles caractéristiques au DataFrame
    data['game_seconds'] = data_milestone_1['prdTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    data['shot_distance'] = distance_goal(data_milestone_1['coord_x'], data_milestone_1['coord_y'])
    data['shot_angle'] = data_milestone_1.apply(lambda row: angle_goal(row['coord_x'], row['coord_y']), axis=1)
    data['rebond'] = (data['last_event_type'] == 'shot-on-goal')
    previous_shot_angle = angle_goal(data['last_event_x'], data['last_event_y'])
    data['changement_angle_tir'] = np.where(data['rebond'], previous_shot_angle + data['shot_angle'], 0)
    data['vitesse'] = data['distance_from_last_event'] / data['time_since_last_event']
    
    # Gestion des cas où time_since_last_event est zéro pour éviter une division par zéro
    data['vitesse'].replace(np.inf, 0, inplace=True)
    data['vitesse'].fillna(0, inplace=True)

    return data


# Fonction pour calculer la distance euclidienne entre deux points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_previous_event(play_data, current_index, current_period, current_period_time):
    if current_index == 0:
        return None
    
    """for i in range(current_index - 1, -1, -1):

        if play_data[i]['typeDescKey'] in ["shot-on-goal", "goal"]:
            previous_index = i
            break"""

    prev_event = play_data[ current_index -1]
    prev_event_period = prev_event['periodDescriptor']['number']
    prev_event_period_time = prev_event['timeInPeriod']

    current_period_time_seconds = int(current_period_time.split(':')[0]) * 60 + int(current_period_time.split(':')[1])
    prev_event_period_time_seconds = int(prev_event_period_time.split(':')[0]) * 60 + int(prev_event_period_time.split(':')[1])

    # Calcul le temps écoulé entre les événements
    if current_period == prev_event_period:
        time_since_last_event = current_period_time_seconds - prev_event_period_time_seconds
    else:
        # Calcul le temps restant dans la période précédente et ajout le temps écoulé dans la période actuelle
        # Ceci suppose que chaque période est de 20 minutes. 
        # Peut-être à ajuster si nécessaire pour les périodes supplémentaires / prolongations.
        time_since_last_event = (20 * 60 - prev_event_period_time_seconds) + current_period_time_seconds
    if 'details' in prev_event:
        last_event_x = prev_event['details'].get('xCoord', pd.NA)
        last_event_y = prev_event['details'].get('yCoord', pd.NA)
    else:
        last_event_x = pd.NA
        last_event_y = pd.NA    
    prev_event_data = {
        'last_event_type': prev_event['typeDescKey'],
        'last_event_x': last_event_x,
        'last_event_y': last_event_y,
        'time_since_last_event': time_since_last_event if time_since_last_event > 0 else pd.NA,
        'distance_from_last_event': pd.NA
    }
    return prev_event_data


def handle_goal_event(play_data, current_index, power_play_status, scoring_team_id):
    """
    Gère les événements de but pour vérifier et annuler les pénalités mineures.
    """

    conceding_team = 'home_team' if scoring_team_id != play_data[current_index]['details']['eventOwnerTeamId'] else 'away_team'

    if power_play_status[conceding_team]:
        power_play_status[conceding_team].sort(key=lambda x: x['start_time'])

        for penalty in power_play_status[conceding_team]:
            if penalty['duration'] == 120:  # 2 minutes en secondes
                power_play_status[conceding_team].remove(penalty)
                break


def update_power_play_status(play_data, current_index, power_play_status, home_team_id, away_team_id):
    current_event = play_data[current_index]
    event_type = current_event['typeDescKey']
    
    current_period_time = int(current_event['timeInPeriod'].split(':')[0]) * 60 + int(current_event['timeInPeriod'].split(':')[1])

    if event_type == "goal" :
        scoring_team_id = current_event['details']['eventOwnerTeamId']
        handle_goal_event(play_data, current_index, power_play_status, scoring_team_id)

    if event_type == 'penalty' :
        team_id = current_event['details']['eventOwnerTeamId']
        penalty_duration = current_event['details']['duration']
        penalized_team = 'home_team' if team_id == home_team_id else 'away_team'

        start_time = int(current_period_time)
        #penalty_duration = current_event['result']['penaltyMinutes']

        if power_play_status[penalized_team] is None:
            power_play_status[penalized_team] = [{'start_time': start_time, 'duration': penalty_duration * 60}]
        else:
            power_play_status[penalized_team].append({'start_time': start_time, 'duration': penalty_duration * 60})
            
    for team, penalties in power_play_status.items():
        if penalties:
            power_play_status[team] = [p for p in penalties if current_period_time - p['start_time'] < p['duration']]
    

def calculate_skater_count(power_play_status, home_team_name, away_team_name):
    standard_skater_count = 5
    home_team_skaters = standard_skater_count
    away_team_skaters = standard_skater_count

    for team_name, penalties in power_play_status.items():
        if penalties:  # Si l'équipe a des pénalités
            penalty_count = len(penalties)
            if team_name == 'home_team':
                home_team_skaters = max(3, standard_skater_count - penalty_count)
            else:
                away_team_skaters = max(3, standard_skater_count - penalty_count)

    return {
        home_team_name: home_team_skaters,
        away_team_name: away_team_skaters
    }
    
    
def transformEventData(df: pd.DataFrame) -> pd.DataFrame:
  
    temp_data = {
        'gameId': [], 'evt_idx': [], 'prd': [], 'prdTime': [],'team': [],
        'coord_x': [], 'coord_y': [],'shotCategory': [],'goalFlag': [],'shotBy': [],
        'goalieName': [], 'last_event_type': [], 'last_event_x': [], 'last_event_y': [],
        'time_since_last_event': [], 'distance_from_last_event': [],
        'power_play_time_elapsed': [], 'home_team_skater_count': [],
        'away_team_skater_count': [],'visitorTeam': [], 'hostTeam': [], 'homeRinkSide': [], 'awayRinkSide': []
    }
    

    for idx in range(df.shape[1]):
        play_data = df.iloc[:, idx]["plays"]
        game_details = df.iloc[:, idx]
        home_team_name = game_details["homeTeam"]["id"]
        away_team_name = game_details["awayTeam"]["id"]
        power_play_status = {'home_team': None, 'away_team': None}
        rink_sides = {}

        for event_index, single_event in enumerate(play_data):

            if single_event['typeDescKey'] not in ["shot-on-goal", "goal"]:
                continue
        
            details = single_event.get("details", {})
            x_coord = details.get("xCoord")
            team_id = details.get("eventOwnerTeamId")
            period = single_event['periodDescriptor']['number']

            if team_id is not None and x_coord is not None:
                if period not in rink_sides:
                    # Assign rinkside based on the first event
                    if x_coord < 0:
                        
                        home_rink_side = "left" if team_id == home_team_name else "right"
                        away_rink_side = "right" if team_id == home_team_name else "left"
                        
                    elif x_coord > 0:

                        home_rink_side = "right" if team_id == home_team_name else "left"
                        away_rink_side = "left" if team_id == home_team_name else "right"
                    if single_event['periodDescriptor']['number'] == 2:
                        rink_sides[single_event['periodDescriptor']['number']] = {"home": away_rink_side, "away": home_rink_side}  # Switch sides in 2nd period
                    else:
                        rink_sides[single_event['periodDescriptor']['number']] = {"home": home_rink_side, "away": away_rink_side}
            rink_sides[single_event['periodDescriptor']['number']] = {"home": home_rink_side, "away": away_rink_side}
            period_time_str = single_event['timeInPeriod']
            minutes, seconds = map(int, period_time_str.split(':'))
            period_time= int(minutes * 60 + seconds)

            update_power_play_status(play_data, event_index, power_play_status, home_team_name, away_team_name)

            temp_data['gameId'].append(game_details.name)
            temp_data['shotCategory'].append(single_event['details'].get('shotType', pd.NA))
            temp_data['coord_x'].append(single_event['details'].get('xCoord', pd.NA))
            temp_data['coord_y'].append(single_event['details'].get('yCoord', pd.NA))
            temp_data['prd'].append(single_event['periodDescriptor']['number'])
            temp_data['evt_idx'].append(single_event['eventId'])
            temp_data['prdTime'].append(single_event['timeInPeriod'])
            if single_event['details']['eventOwnerTeamId'] == home_team_name:

                temp_data['team'].append(df.iloc[:, idx]['homeTeam']['commonName']['default'])

            else : 
                temp_data['team'].append(df.iloc[:, idx]['awayTeam']['commonName']['default'])

            temp_data['goalFlag'].append(single_event['typeDescKey'] == "goal")
            temp_data['homeRinkSide'].append(rink_sides[period]["home"]) # à vérifier
            temp_data['awayRinkSide'].append(rink_sides[period]["away"])
            #str_code = 'NA' if evt_type == "SHOT" else single_event['result']['strength']['code']
            #temp_data['teamStrength'].append(str_code)
            
            temp_data['visitorTeam'].append(f'{game_details['awayTeam']['commonName']['default']}, {game_details['awayTeam']['abbrev']}')
            temp_data['hostTeam'].append(f'{game_details['homeTeam']['commonName']['default']}, {game_details['homeTeam']['abbrev']}')
            temp_data['shotBy'].append(single_event['details'].get('shootingPlayerId', pd.NA))
            temp_data['goalieName'].append(single_event['details'].get('goalieInNetId', pd.NA))


            # Ajout des données de l'événement précédent
            current_period = single_event['periodDescriptor']['number']
            current_period_time = single_event['timeInPeriod']
            
                        
            prev_event_data = find_previous_event(play_data, event_index, current_period, current_period_time)

            if prev_event_data:
                #coordonnées du tir actuel pour calculer la distance
                current_x = single_event['details'].get('xCoord', pd.NA)
                current_y = single_event['details'].get('yCoord', pd.NA)

                # Calcul des données temporelles et spatiales si les données sont complètes
                if prev_event_data['last_event_x'] is not pd.NA and prev_event_data['last_event_y'] is not pd.NA and current_x is not pd.NA and current_y is not pd.NA:
                    prev_event_data['distance_from_last_event'] = calculate_distance(
                        prev_event_data['last_event_x'], prev_event_data['last_event_y'],
                        current_x, current_y)
            
                for key in ['last_event_type', 'last_event_x', 'last_event_y', 'time_since_last_event', 'distance_from_last_event']:
                    temp_data[key].append(prev_event_data.get(key, pd.NA))
            else:
                # Si prev_event_data est None (c'est le premier événement), on ajoute des valeurs NA.
                for key in ['last_event_type', 'last_event_x', 'last_event_y', 'time_since_last_event', 'distance_from_last_event']:
                    temp_data[key].append(pd.NA)
                    
            elapsed_time = 0
            for status in power_play_status.values():
                if status:
                    for penalty in status:
                        elapsed_time = max(elapsed_time, period_time - penalty['start_time'])
                        
            temp_data['power_play_time_elapsed'].append(elapsed_time)

            skater_count = calculate_skater_count(power_play_status, home_team_name, away_team_name)
            temp_data['home_team_skater_count'].append(skater_count[home_team_name])
            temp_data['away_team_skater_count'].append(skater_count[away_team_name])
            
    output_df = pd.DataFrame(temp_data)
    #output_df['coord_x'] = pd.to_numeric(output_df['coord_x'], errors='coerce')
    #output_df['coord_y'] = pd.to_numeric(output_df['coord_y'], errors='coerce')
    #output_df['last_event_x'] = pd.to_numeric(output_df['last_event_x'], errors='coerce')
    #output_df['last_event_y'] = pd.to_numeric(output_df['last_event_y'], errors='coerce')
    output_df['game_seconds'] = output_df['prdTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    output_df['shot_distance'] = output_df.apply(lambda row: distance_goal(row['coord_x'], row['coord_y']), axis=1)
    output_df['shot_angle'] = output_df.apply(lambda row: angle_goal(row['coord_x'], row['coord_y']), axis=1)
    output_df['rebond'] = (output_df['last_event_type'] == 'shot-on-goal')
    previous_shot_angle = output_df.apply(lambda row: angle_goal(row['last_event_x'], row['last_event_y']), axis=1)
    output_df['changement_angle_tir'] = np.where(output_df['rebond'], previous_shot_angle + output_df['shot_angle'], 0)
    output_df['vitesse'] = output_df['distance_from_last_event'] / output_df['time_since_last_event']
    
    # Gestion des cas où time_since_last_event est zéro pour éviter une division par zéro
    output_df['vitesse'].replace(np.inf, 0, inplace=True)
    output_df['vitesse'].fillna(0, inplace=True)
    folder_path = '../data/derivatives'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save the dataframe to the specified path
    output_df.to_csv(os.path.join(folder_path, 'DATA.csv'), index=False)
    return output_df

def fusion_features(engineering1, engineering2):
    
    if len(engineering1) != len(engineering2):
        raise ValueError("Les DataFrames doivent avoir le même nombre de lignes")
    
    #engineering2['distance_shot'] = engineering1['distance_goal']
    #engineering2['angle_shot'] = engineering1['angle_goal']
    engineering2['is_goal'] = engineering1['is_goal']
    #engineering2['empty_net'] = engineering1['empty_goal']  
        
    return engineering2

def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    except ValueError:
        return None  


def preprocessing(df: pd.DataFrame, target: str):
    """
    Fonction pour prétraiter les données avant de les utiliser dans un modèle. Cette fonction
    permet de transformer les variables catégorielles en variables numériques et de normaliser
    les variables numériques.

    Parameters
    ----------
    df: DataFrame
        Caractéristiques à prétraiter
    target: str
        Nom de la variable cible

    Returns
    -------
    X: DataFrame
        Données prétraitées
    y: DataFrame
        Variable cible
    """

    # On supprime les colonnes avec plus de 50% de NaN
    half_count = len(df) / 2
    df = df.dropna(thresh=half_count, axis=1)
    df = df.dropna(subset=['shot_distance', 'shot_angle'])
    # Supprime les lignes avec des NaN
    #df = df.dropna()
    df = df.drop(columns=['gameId'])
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    df['shotBy'] = df['shotBy'].fillna("Unknown")

    # 2. Fill 'goalieName' with "Unknown"
    df['goalieName'] = df['goalieName'].fillna(8888888.0)
    # Colonnes One-Hot Encoding
    """cols_to_encode = ['shotCategory', 'last_event_type','team', 
        'visitorTeam', 'hostTeam', 'homeRinkSide', 'awayRinkSide']
    df_encoded = pd.get_dummies(df[cols_to_encode], dtype=int)
    df = pd.concat([df, df_encoded], axis=1).drop(cols_to_encode, axis=1)"""
    # Colonnes à binariser
    cols_to_binarize = ['rebond', 'shotCategory', 'last_event_type','team', 
        'visitorTeam', 'hostTeam', 'homeRinkSide', 'awayRinkSide']
    for col in cols_to_binarize:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    

    # Extraction de la variable cible
    y = df[target]
    # Binarisation de la variable cible
    y = LabelEncoder().fit_transform(y)
    # Extraction des caractéristiques
    X = df.drop(target, axis=1)


    # 3. Fill 'last_event_x', 'last_event_y', and 'distance_from_last_event' with the mean
    X['last_event_x'] = X['last_event_x'].fillna(X['last_event_x'].mean())
    X['last_event_y'] = X['last_event_y'].fillna(X['last_event_y'].mean())
    X['distance_from_last_event'] = X['distance_from_last_event'].fillna(X['distance_from_last_event'].mean())
    X['changement_angle_tir'] = X['changement_angle_tir'].fillna(X['changement_angle_tir'].mean())

    # 4. Fill 'time_since_last_event' with the median
    X['time_since_last_event'] = X['time_since_last_event'].fillna(X['time_since_last_event'].median())
    # Apply conversion
    X["prdTime"] = X["prdTime"].apply(convert_time_to_seconds)
    # Standardisation des variables numériques
    cols_to_standardize = [
        'coord_x', 'coord_y', 'last_event_x', 'last_event_y',  
        'time_since_last_event', 'distance_from_last_event', 
        'game_seconds', 'shot_distance', 'shot_angle', 
        'changement_angle_tir', 'vitesse', 'power_play_time_elapsed', 
        'home_team_skater_count', 'away_team_skater_count']
    
    features = X[cols_to_standardize]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    X[cols_to_standardize] = features

    return X, y

def comet_log_dataframe_profile():
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='Milestone_2',
        workspace='me-pic',
    )

    experiment.set_name("Dataframe")

    data = pd.read_csv('../data/derivatives/dataframe_milestone_2.csv')
    
    subset_df = data[data['gameId'] == 2017021065]

    experiment.log_dataframe_profile(
        subset_df,
        name='wpg_v_wsh_2017021065',
        dataframe_format='csv'
    )

    experiment.end()
