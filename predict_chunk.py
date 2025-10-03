# #predict_chunk.py

from audio_search_main import AudioSearch, load_audio_data
from mongo_audio_print_db import MongoAudioPrintDB
import numpy as np
import pandas as pd

# ---- Confidence thresholds (tune for telephony honeypots) ----
MIN_TOP_BIN = 20           # require at least 20 hashes in the top offset bin
MIN_RATIO = 2.0            # top bin must be at least 2x stronger than runner-up
MIN_TOTAL_MATCHES = 60     # require at least 60 matching hashes overall
MIN_UNIQUE_OFFSETS = 4     # require ≥4 distinct offset bins for the winner

def pick_winner_with_confidence(df_matches, n_fingerprints):
    """Decide winner or None based on confidence gating."""
    if df_matches is None or df_matches.empty:
        return None, {"reason": "no matches"}

    if not isinstance(df_matches, pd.DataFrame):
        return None, {"reason": "bad df"}

    # Ensure df has songID and offset columns
    if 'songID' not in df_matches.columns:
        # sometimes songID may be index, reset it
        df_matches = df_matches.reset_index()

    if 'stk' not in df_matches.columns:
        # if offset also lives in the index, reset
        df_matches = df_matches.reset_index()

    # Count matches per (songID, offset)
    g = df_matches.groupby(['songID', 'stk']).size().reset_index(name='n')

    # Sort by strength
    g_sorted = g.sort_values('n', ascending=False)
    top = g_sorted.iloc[0]
    top_sid, top_bin, top_offset = top['songID'], int(top['n']), int(top['stk'])
    runner_bin = int(g_sorted.iloc[1]['n']) if len(g_sorted) > 1 else 0

    # Per-song totals
    per_song = g.groupby('songID')['n'].sum()
    top_total = int(per_song.loc[top_sid])
    unique_offsets = int(g[g['songID'] == top_sid]['stk'].nunique())

    # Apply thresholds
    passes = True
    reasons = []
    if top_bin < MIN_TOP_BIN:
        passes = False; reasons.append(f"top_bin<{MIN_TOP_BIN}")
    if runner_bin and (top_bin / max(1, runner_bin) < MIN_RATIO):
        passes = False; reasons.append(f"ratio<{MIN_RATIO}")
    if top_total < MIN_TOTAL_MATCHES:
        passes = False; reasons.append(f"total<{MIN_TOTAL_MATCHES}")
    if unique_offsets < MIN_UNIQUE_OFFSETS:
        passes = False; reasons.append(f"unique<{MIN_UNIQUE_OFFSETS}")

    dbg = {
        "top_sid": int(top_sid),
        "top_bin": top_bin,
        "runner_bin": runner_bin,
        "top_total": top_total,
        "unique_offsets": unique_offsets,
        "n_fingerprints": n_fingerprints,
        "passed": passes,
        "fail_reasons": reasons
    }

    return (top_sid if passes else None), dbg

def main(chunk_file):
    db = MongoAudioPrintDB()
    search = AudioSearch(audio_prints_db=db, do_plotting=False)

    # chunk_file = "/Users/iqbabar/Downloads/data1/8k-mono-mp3/honeypot2_honeypot2_segment_001.mp3"
    data, rate = load_audio_data(chunk_file)

    fingerprints = search.get_fingerprints_from_audio(data, rate)
    print(f"Generated {len(fingerprints)} fingerprints")

    if not fingerprints:
        print("❌ No fingerprints generated")
        return

    df_matches = search.get_df_of_fingerprint_offsets(fingerprints)
    if df_matches.empty:
        print("❌ No matches in DB")
        return

    winner, dbg = pick_winner_with_confidence(df_matches, len(fingerprints))
    if winner is None:
        print("❌ Reject (low confidence)", dbg)
    else:
        song_doc = db.find_one_song({'_id': winner})
        print("✅ Confident match:", song_doc, "| metrics:", dbg)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch predict audio chunks against MongoDB fingerprints")
    parser.add_argument("chunk_file", help="File .mp3/.wav chunks to predict")
    args = parser.parse_args()

    main(args.chunk_file)




# from audio_search_main import AudioSearch, load_audio_data
# from mongo_audio_print_db import MongoAudioPrintDB

# db = MongoAudioPrintDB()
# search = AudioSearch(audio_prints_db=db, do_plotting=False)

# chunk_file = "/Users/iqbabar/Downloads/data1/8k-mono-mp3/honeypot2_honeypot2_segment_001.mp3"
# data, rate = load_audio_data(chunk_file)

# fingerprints = search.get_fingerprints_from_audio(data, rate)
# df_matches = search.get_df_of_fingerprint_offsets(fingerprints)

# if df_matches.empty:
#     print({404})
# else:
#     index_set = set(df_matches.index)
#     n_possible_songs = len(index_set)
#     winner = search.get_the_most_likely_song_from_all_the_histograms(
#         df_matches, n_possible_songs, index_set
#     )
#     song_doc = db.find_one_song({'_id': winner})
#     print("✅ Match found:", song_doc)

