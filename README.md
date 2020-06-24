## Usage

### Sending a json object containing the"track_key" that looks like:
```sh
{
  "track_key": 1
}
```

to `http://127.0.0.1:5000/recommend`

will return an jsonified list of track_ids of the closest related 10songs from songs.csv

Output:
```sh
[
    {
        "artist_name": "YG",
        "track_id": "2RM4jf1Xa9zPgMGRDiht8O",
        "track_name": "Big Bank feat. 2 Chainz, Big Sean, Nicki Minaj"
    },
    {
        "artist_name": "Gyyps",
        "track_id": "02s1Voowwhr0qTSOrMVEXk",
        "track_name": "PRETEND"
    },
    {
        "artist_name": "Yori",
        "track_id": "6PIdcDdMuoyAWeVqMRMLlR",
        "track_name": "Glitter Mom"
    },
    {
        "artist_name": "Ca$his",
        "track_id": "0MiAP4EQGIuikH23RTyP53",
        "track_name": "What"
    },
    {
        "artist_name": "Caash",
        "track_id": "2lD4dQpcac8H2Rq5nttJJh",
        "track_name": "Bands"
    },
    {
        "artist_name": "Joey Trap",
        "track_id": "07lPxulz2gpCAVvgemzedl",
        "track_name": "Grace"
    },
    {
        "artist_name": "Joe Blow",
        "track_id": "5MccaBOgJh4MsJNoj72eij",
        "track_name": "Killing Spree (feat. Ampichino, Twisted Revren, 38 Spesh & Shredgang Mone)"
    },
    {
        "artist_name": "Brose Royce",
        "track_id": "3QBTZEqaCbJxztWh5NTKHs",
        "track_name": "Never Been a Fraud"
    },
    {
        "artist_name": "24hrs",
        "track_id": "6yoM6KoDpWXKIWo9UnZ3Ef",
        "track_name": "DRUGS"
    },
    {
        "artist_name": "Blac Youngsta",
        "track_id": "7Jbt6KWzS8cpP1xnW0tQ96",
        "track_name": "Beginners"
    }
]
```

### Sending 
```sh
{
  "track_key": 1
}
```

to `http://127.0.0.1:5000/recommend`

will return a list of averaged features for the closest related 10songs for visualizations.

Output:
```sh
[
    0.06,
    0.73,
    0.42,
    0.0,
    0.12,
    1.0,
    0.41,
    0.25
]
```