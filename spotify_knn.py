import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

CSV_PATH = "songs.csv"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# select numeric features
FEATURES = [
    "artist_count",
    "released_year",
    "released_month",
    "released_day",
    "in_spotify_playlists",
    "in_spotify_charts",
    "streams",
    "in_apple_playlists",
    "in_apple_charts",
    "in_deezer_playlists",
    "in_deezer_charts",
    "in_shazam_charts",
    "bpm",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%"
]

# make sure columns exist
for col in FEATURES:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")

# ensure numeric
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")

# drop rows with missing feature data
df = df.dropna(subset=FEATURES)

X = df[FEATURES]


# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train model
K = 10
model = NearestNeighbors(n_neighbors=K, metric="euclidean")
model.fit(X_scaled)

print("\nKNN model trained.\n")

# recommend function
def recommend(song_name, k=10):
    # find song by name 
    matches = df[df["track_name"].str.lower() == song_name.lower()]

    if matches.empty:
        print(f"No song named '{song_name}' found.")
        return

    song = matches.iloc[0]
    print("\nQuery:", song["track_name"], "-", song["artist(s)_name"])

    song_vec = song[FEATURES].values.reshape(1, -1)
    song_vec_scaled = scaler.transform(song_vec)

    distances, indices = model.kneighbors(song_vec_scaled, n_neighbors=k + 1)

    # first index is the song itself
    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]

    recs = df.iloc[rec_indices].copy()
    recs["distance"] = rec_distances

    cols_to_show = ["track_name", "artist(s)_name", "distance"]
    print("\nTop recommendations:\n")
    print(recs[cols_to_show].reset_index(drop=True))

# CLI loop
if __name__ == "__main__":
    print("Spotify KNN recommender using songs.csv")
    while True:
        name = input("\nEnter song title (or press Enter to quit): ").strip()
        if name == "":
            break
        recommend(name)
