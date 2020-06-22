import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from flask import jsonify

# load data
data = pd.read_csv(
    "https://raw.githubusercontent.com/Lambda-Spotify-Song-Suggester-3/datascience/master/kaggle_data/encoded.csv")
df = data.copy()

dictionary = df[["artist_name", "track_name", "track_key", "track_id"]]
dictionary_no_track_key = df[["artist_name", "track_name", "track_id"]]
# drop columns for training
df = df.drop(
    columns=[
        'artist_name',
        'track_id',
        'track_name',
        'track_key',
        'duration_ms',
        'mode',
        'loudness',
        'time_signature'])

# Scale the data

scaler = StandardScaler()
df_s = scaler.fit_transform(df)


def predictor(track_key):
    '''
    Function to take "track key" of a song of interest from dataframe, and
    return a list of track_ids of the closest related 10songs.
    output format:
  [
    {
        "artist_name": "French Montana",
        "track_id": "2RC86WM8zX8v1z4QNtZ2sI",
        "track_name": "New Goals (French Montana)"
    },
    '''

    # Convert "track_key" to the index of the song
    input_dictionary_entry = dictionary[dictionary['track_key'] == track_key]
    input_index = input_dictionary_entry.index[0]

    # Nearest Neighbors model
    nn = NearestNeighbors(n_neighbors=10, algorithm='kd_tree')
    nn.fit(df_s)

    neighbor_predictions = nn.kneighbors([df_s[input_index]])

    # This is a list of the INDEXES of the songs
    list_of_predictions = neighbor_predictions[1][0].tolist()

    similar_tracks_keys = []
    for item in list_of_predictions:
        track_hash = dictionary["track_id"].iloc[item]
        similar_tracks_keys.append(track_hash)

    similar_tracks_no_keys = []
    for val in list_of_predictions:
        entries = dict(dictionary_no_track_key.iloc[val])
        similar_tracks_no_keys.append(entries)

    return similar_tracks_keys, similar_tracks_no_keys


# testing functionality
# print(predictor(1))


# song features, for plotting
def feature_average(track_key):
    '''
    This function returns the sum of the features for the ten recommended songs.
    '''
    similar_tracks = predictor(track_key)
    # Return a dataframe with only the ten most similar tracks
    similar_tracks = data[data["track_id"].isin(similar_tracks[0])]
    similar_tracks = similar_tracks[['acousticness', 'danceability',
                                     'energy', 'instrumentalness',
                                     'liveness', 'mode',
                                     'speechiness', 'valence']]
    # Average features of ten tracks
    acousticness = round(similar_tracks['acousticness'].mean(), 2)
    danceability = round(similar_tracks['danceability'].mean(), 2)
    energy = round(similar_tracks['energy'].mean(), 2)
    instrumentalness = round(similar_tracks['instrumentalness'].mean(), 2)
    liveness = round(similar_tracks['liveness'].mean(), 2)
    mode = round(similar_tracks['mode'].mean(), 2)
    speechiness = round(similar_tracks['speechiness'].mean(), 2)
    valence = round(similar_tracks['valence'].mean(), 2)
    # Store all to "features" variable
    features = []
    attributes = [
        acousticness,
        danceability,
        energy,
        instrumentalness,
        liveness,
        mode,
        speechiness,
        valence]
    # features.append(acousticness)
    for attribute in attributes:
        features.append(attribute)
    return features


# # testing functionality
# print(feature_average(1))