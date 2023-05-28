import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time
import json
import yaml
import numpy as np 
from numpy import reshape
import scipy.stats as st
import re
from collections import Counter
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF



def login(spotify_client_id, spotify_client_secret, openai_api_key):
   spotify_credentials = SpotifyClientCredentials(client_id=spotify_client_id, 
                                                  client_secret=spotify_client_secret)
   spotify_client = spotipy.Spotify(client_credentials_manager=spotify_credentials)
   openai.api_key = openai_api_key
   return spotify_client

CHAT_CACHE = {}
def chat(messages=[], model="gpt-3.5-turbo"):
   try:
      key = model+str(messages)
      if key in CHAT_CACHE:
        response = CHAT_CACHE[key]
      else:
        response =  openai.ChatCompletion.create(model=model, messages=messages)
        CHAT_CACHE[key] = response
      answer = response["choices"][0]["message"]["content"]
      usage = response["usage"]["total_tokens"]
      return answer, usage
   except openai.error.RateLimitError as e:
      retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
      print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      return chat(messages, model)

def system_msg(str):
  return  {"role": "system", "content": str}
def assistant_msg(str):
  return  {"role": "assistant", "content": str}
def user_msg(str):
  return  {"role": "user", "content": str}



def songs_by_vibe(vibe,  model= "gpt-4"): #"gpt-3.5-turbo" doesnt seem to work well.
  songs, tokens = chat([system_msg("Always format the result as JSON."), user_msg(f"Give me 10 songs and their artists that have the vibe of {vibe}")],  model=model)
  if model != "gpt-4":
    print 
    songlist = []
    rows = songs.split("\n")
    for row in rows:
      parts = row.split("\"")
      if len(parts) != 3:
        continue
      song = {}
      song['title'] = parts[1]
      song['artist'] = re.sub("^ (-|by) ","",parts[2])
      songlist.append(song)
    return songlist
  try:
    return json.loads(songs)['songs']
  except:
    return []

def pp(results):
  array = results['tracks']['items']
  for i, subdict in enumerate(array):
    print(f"TRACK {i}: ")
    for k in subdict:
      if k == 'available_markets':
        continue
      if k == "album":
        del(subdict[k]['available_markets'])
      print(f"{k.upper()}: {yaml.dump(subdict[k], default_flow_style=True)}")

def lookup_songs(spotify_client, songs):
  tracks = []
  for song in songs:
    artist = song['artist']
    if 'title' in song:
      track = song['title']
    elif 'song' in song:
      track = song['song']
    else:
      raise f"{song} does not have a title."
    q = f"track:{track} artist:{artist}"
    song['url'] = "NOTFOUND"
    song['uri'] = "NOTFOUND"
    try:
      results = spotify_client.search(q, limit=1, offset=0, type='track', market=None)
      song['url'] = results['tracks']['items'][0]['external_urls']
      song['uri'] = results['tracks']['items'][0]['uri']
      song['artist_uri'] = results['tracks']['items'][0]['artists'][0]['uri']
      try:
        song['artist_genres'] = spotify_client.artist(song['artist_uri'])["genres"]
      except:
        song['artist_genres'] = []
    except:
      time.sleep(1.0)
    tracks.append(song)
  return tracks

def get_and_set_features(spotify_client, songs):
  track_uris = [song['uri'] for song in songs]
  features = spotify_client.audio_features(track_uris)
  for f in features:
    try:
      uri = f['uri']
    except:
      continue
    for del_attr in ['uri', 'analysis_url', 'track_href', 'type', 'id']:
      del(f[del_attr])
    for song in songs:
      if song['uri'] == uri:
        song['features'] = f
        #print(f"{song['title'].upper()}: {yaml.dump(song['features'], default_flow_style=True)}")

FEATURES = ['acousticness', 'danceability', 'energy', 
            'instrumentalness','loudness', 'valence', 'tempo', 'key']

def get_features():
  return FEATURES

def make_top_genre_list(validated_songs, n=-1):
  c = Counter()
  for s in validated_songs:
    if 'artist_genres' in s:
      c.update(Counter(s['artist_genres']))
  if n == -1:
    return c
  return(c.most_common(n=n))

def make_farray(validated_songs, label, features=FEATURES):
  #sorted(validated_songs2[0]['features'].keys())
  feature_list = FEATURES
  target = []
  data = []
  songstrs = []
  for song in validated_songs:
    if 'features' in song:
      flist = []
      for f in feature_list:
        if f in song['features']:
         flist.append(song['features'][f])
        else:
          flist.append(0.0)
      data.append(flist)
      target.append(label)
      songstrs.append(f"{song['title']} by {song['artist']}")
  return target, data, songstrs

def print_songs(validated_songs):
  for song in validated_songs:
    try:
      print(f"[{song['title']}] BY [{song['artist']}] [{song['url']}] [{song['features']}]")
    except:
      print(f"[{song['title']}] BY [{song['artist']}] - some data missing from spotify")

def get_recommendations_one_shot(spotify_client, validated_songs):
  track_uris = [song['uri'] for song in validated_songs if song['uri'] != 'NOTFOUND']
  reccos=spotify_client.recommendations(seed_tracks=track_uris[0:5], limit=50)
  new_songs = []
  for track in reccos['tracks']:
    new_songs.append({
           'title': track['name'],
           'artist': track['artists'][0]['name'],
           'url' : track['external_urls'],
           'uri' : track['uri']})
  return new_songs

#use permutations?
def get_recommendations(spotify_client, validated_songs, n=5):
  new_songs = []
  track_uris = [song['uri'] for song in validated_songs if song['uri'] != 'NOTFOUND']
  for track in track_uris:
    reccos=spotify_client.recommendations(seed_tracks=[track], limit=n)
    for track in reccos['tracks']:
      new_songs.append({
           'title': track['name'],
           'artist': track['artists'][0]['name'],
           'url' : track['external_urls'],
           'uri' : track['uri']})
  # and de-dupe?
  return new_songs

def get_recommendations_by_vibe(spotify_client, vibe,  features=FEATURES, model="gpt-4", recommendations=True):
  songs = songs_by_vibe(vibe,model=model)
  validated_songs = lookup_songs(spotify_client, songs)
  get_and_set_features(spotify_client, validated_songs)
  target, data, songstr = make_farray(validated_songs, vibe, features)
  if not recommendations:
    return validated_songs, target, data, songstr, [], [], [], []
  recommended_songs = get_recommendations(spotify_client, validated_songs)
  get_and_set_features(spotify_client, recommended_songs)
  recco_target, recco_data, recco_songstr = make_farray(recommended_songs,  "recommended: "+ vibe, features)
  return validated_songs, target, data, songstr, recommended_songs, recco_target, recco_data, recco_songstr

def songs_value_range(songs,ignore = []):
  f_list = {}
  for song in songs:
    if 'features' not in song:
      continue
    for f in song['features']:
      if f in ignore:
        continue
      if f not in f_list:
        f_list[f] = []
      f_list[f].append(song['features'][f])
  for f in f_list:
    data = f_list[f]
    f_list[f] = np.around(st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)), 2)
  return f_list

#based off of https://mclguide.readthedocs.io/en/latest/sklearn/clusterdim.html


def pca(list_of_vibe_data):
  """ 
    expect a list of len N vibe data. Each vibe data has 8 points:
     validated_songs, data, target and strings.
     recommended_songs, data, target and strings
  """

  num_of_vibes =  len(list_of_vibe_data)
  
  x =[]
  y = []
  songnames = []
  for vibe_data in list_of_vibe_data:
   
  
    y += vibe_data[1] + vibe_data[5]
    x += vibe_data[2] + vibe_data[6]
    songnames +=  vibe_data[3] + vibe_data[7]

 
  x=np.array(x)
  y=np.array(y) 
  songnames=np.array(songnames)


  # Normalize Features to -1, 1, then Normalize Members to unit length, the
  # Dimesionality reduction to 2
  T0 = preprocessing.RobustScaler().fit_transform(x)
  T1 = preprocessing.Normalizer().fit_transform(T0)
  pca_model = PCA(n_components=2)
  T2 = pca_model.fit_transform(T1) # fit the model


  # store the values of PCA component in variable: for easy writing
  xvector = pca_model.components_[0] * max(T2[:,0])
  yvector = pca_model.components_[1] * max(T2[:,1])

  df = pd.DataFrame(x, columns = chatspot.get_features())
  df["vibe"] = y
  df["songnames"] = songnames
  df["comp-1"] = T2[:, 0]
  df["comp-2"] = T2[:, 1]


  rbf = GaussianProcessClassifier(1.0 * RBF(1.0))
  rbf.fit(T2,y)

  x_min, x_max = -1, 1.1
  y_min, y_max =-1, 1.1

  h = .01
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  y_ = np.arange(y_min, y_max, h)
  Z = rbf.predict(np.c_[xx.ravel(), yy.ravel()])

  label_dict = {}
  n_colors = 1
  for label in y:
    if label not in label_dict:
      label_dict[label] = n_colors - 0.1
      n_colors += 1

  Z = np.array([label_dict[ele] for ele in Z])
  Z = Z.reshape(xx.shape)

  # Don't ask me why this works... at least for 4 colors
  colormap = px.colors.sample_colorscale("rainbow", [n/(n_colors -1) for n in range(n_colors)])[1:]
  heatmap = go.Heatmap(x=xx[0],
                    y=y_, 
                    z=Z,
                    zmin=0,
                    zmax=(n_colors-1),
                    hoverinfo='skip',
                    colorscale="rainbow",
                    showscale=False)


  title='Spotify Feature PCA projection'
  hover_data={'vibe': False, 'comp-1':False, 'comp-2':False}
  for f in chatspot.get_features():
    hover_data[f]=True

  fig = px.scatter(df, hover_name='songnames', hover_data=hover_data, title=title,
                 x="comp-1", y="comp-2", color='vibe', width=1000, height=800,
                 color_discrete_sequence=colormap)
  fig.update_traces(marker=dict(size=6, line=dict(width=1, color='black')))
 
  fig.add_trace(heatmap)

  fig.update_layout(title={'font_family': "Arial",
                         'x':0.45,
                         'y': 0.92,    
                         'xanchor': 'center' })


  fig.update_yaxes(tickvals=[], showgrid=False, visible=False)
  fig.update_xaxes(tickvals=[], showgrid=False, visible=False)

  dim_sizes = []
  for i in range(len(chatspot.get_features())):
    dim_sizes.append(math.sqrt(pow(xvector[i],2) + pow(yvector[i],2)))
    dim_sizes = sorted(dim_sizes, reverse=True)
  third = dim_sizes[3]

  arrowsize = 1
  arrowhead = 1
  arrowscale = 1
  for i, feature in enumerate(chatspot.get_features()):
        if math.sqrt(pow(xvector[i],2) + pow(yvector[i],2)) < third:
          continue
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=xvector[i]*arrowscale,
            y=yvector[i]*arrowscale,
            showarrow=True,
            arrowsize=arrowsize,
            arrowhead=arrowhead,
            xanchor="right",
            yanchor="top"
        )
        fig.add_annotation(
            x=xvector[i]*arrowscale,
            y=yvector[i]*arrowscale,
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            bgcolor="white",
            yshift=5,
        )

  return fig
 
