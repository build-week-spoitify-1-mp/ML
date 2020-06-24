import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load Data
data = pd.read_csv("https://raw.githubusercontent.com/Lambda-Spotify-Song-Suggester-3/datascience/master/kaggle_data/encoded.csv")
df = data.copy()

dictionary = df[["artist_name", "track_name", "track_key", "track_id"]]

# Drop
df = df.drop(columns=['artist_name','track_id', 'track_name','track_key', 'duration_ms', 'loudness', 'time_signature'])

scaler = MinMaxScaler()
df_s = scaler.fit_transform(df)

def predictor(track_key):


  # Cosign Similarity
  matrix = cosine_similarity(df_s, df_s[track_key:(track_key + 1)])
  matrix = pd.DataFrame(matrix)
  top = matrix[0].sort_values(ascending=False)[:10]

  # Print Playlist
  z = top.reset_index()
  similar_tracks = []

  for col in z["index"]:
    track = (dictionary['track_id'].iloc[col])
    similar_tracks.append(track)

  return similar_tracks

def feature_average(track_key):
  '''
  This function returns the sum of the features for the ten recommended songs.
  '''
  similar_tracks = predictor(track_key)
  # Return a dataframe with only the ten most similar tracks
  similar_tracks = data[data["track_id"].isin(similar_tracks)]
  similar_tracks = similar_tracks[['acousticness', 'danceability', 
                                           'energy', 'instrumentalness', 
                                           'liveness', 'mode', 
                                           'speechiness', 'valence']]
  # Average features of ten tracks                                           
  acousticness = round(similar_tracks['acousticness'].mean(),2)
  danceability = round(similar_tracks['danceability'].mean(),2)
  energy = round(similar_tracks['energy'].mean(),2)
  instrumentalness = round(similar_tracks['instrumentalness'].mean(),2)
  liveness = round(similar_tracks['liveness'].mean(),2)
  mode = round(similar_tracks['mode'].mean(),2)
  speechiness = round(similar_tracks['speechiness'].mean(),2)
  valence = round(similar_tracks['valence'].mean(),2)
  # Store all to "features" variable
  features = []
  attributes = [acousticness, danceability, energy, instrumentalness, liveness, mode, speechiness, valence]
  #features.append(acousticness)
  for attribute in attributes:
    features.append(attribute)
  return features