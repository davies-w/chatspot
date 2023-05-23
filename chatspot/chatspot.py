import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time
import json
import yaml


def login(spotify_client_id, spotify_client_secret, openai_api_key):
   spotify_credentials = SpotifyClientCredentials(client_id=spotify_client_id, 
                                                  client_secret=spotify_client_secret)
   spotify_client = spotipy.Spotify(client_credentials_manager=spotify_credentials)
   openai.api_key = openai_api_key
   return spotify_client


def chat(messages=[], model="gpt-3.5-turbo"):
   try:
      response =  openai.ChatCompletion.create(
      model=model,
      messages=messages
      )
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
  songs, tokens = chat([system_msg("Always format the result as JSON."), user_msg(f"Give me 10 songs and their artists that have the vibe of {vibe} ")],  model=model)
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

def make_farray(spotify_client, validated_songs, label, features=FEATURES):
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

def get_recommendations_by_vibe(spotify_client, vibe,  features=FEATURES, model="gpt-4"):
  songs = songs_by_vibe(vibe,model=model)
  validated_songs = lookup_songs(spotify_client, songs)
  get_and_set_features(spotify_client, validated_songs)
  target, data, songstr = make_farray(spotify_client, validated_songs, vibe, features)
  recommended_songs = get_recommendations(spotify_client, validated_songs)
  get_and_set_features(spotify_client, recommended_songs)
  recco_target, recco_data, recco_songstr = make_farray(recommended_songs,  "recommended: "+ vibe, features)
  return validated_songs, target, data, songstr, recommended_songs, recco_target, recco_data, recco_songstr
