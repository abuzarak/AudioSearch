# predict_chunk_batch.py
import os
from audio_search_main import AudioSearch, load_audio_data
from mongo_audio_print_db import MongoAudioPrintDB

def predict_directory(directory):
    # connect to Mongo
    db = MongoAudioPrintDB()
    search = AudioSearch(audio_prints_db=db, do_plotting=False)

    # walk directory
    for filename in os.listdir(directory):
        if not filename.lower().endswith((".mp3", ".wav")):
            continue

        file_path = os.path.join(directory, filename)
        print(f"\n🔎 Processing: {file_path}")

        try:
            # load audio
            data, rate = load_audio_data(file_path)
            # fingerprints
            fingerprints = search.get_fingerprints_from_audio(data, rate)
            if not fingerprints:
                print("❌ No fingerprints generated")
                continue

            # match against Mongo
            df_matches = search.get_df_of_fingerprint_offsets(fingerprints)
            if df_matches.empty:
                print("❌ No match found")
                continue

            # determine best match
            index_set = set(df_matches.index)
            n_possible_songs = len(index_set)
            winner = search.get_the_most_likely_song_from_all_the_histograms(
                df_matches, n_possible_songs, index_set
            )
            song_doc = db.find_one_song({'_id': winner})
            print(f"✅ Match found: {song_doc.get('title', 'Unknown')} ({winner})")

        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch predict audio chunks against MongoDB fingerprints")
    parser.add_argument("directory", help="Directory containing .mp3/.wav chunks to predict")
    args = parser.parse_args()

    predict_directory(args.directory)

