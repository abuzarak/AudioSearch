import os
import tempfile
import zipfile
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment

from audio_search_main import AudioSearch, load_audio_data
from mongo_audio_print_db import MongoAudioPrintDB

# ---- Confidence thresholds ----
MIN_TOP_BIN = 10
MIN_RATIO = 2.0
MIN_TOTAL_MATCHES = 60
MIN_UNIQUE_OFFSETS = 4

db = MongoAudioPrintDB()
search = AudioSearch(audio_prints_db=db, do_plotting=False)
app = FastAPI(title="Honeypot Audio Matcher API")

# def pick_winner_with_confidence(df_matches, n_fingerprints):
#     if df_matches is None or df_matches.empty:
#         return None, {"reason": "no matches"}

#     if not isinstance(df_matches, pd.DataFrame):
#         return None, {"reason": "bad df"}

#     if 'songID' not in df_matches.columns:
#         df_matches = df_matches.reset_index()
#     if 'stk' not in df_matches.columns:
#         df_matches = df_matches.reset_index()

#     if 'songID' not in df_matches.columns or 'stk' not in df_matches.columns:
#         return None, {"reason": "missing columns", "cols": list(df_matches.columns)}

#     g = df_matches.groupby(['songID', 'stk']).size().reset_index(name='n')
#     g_sorted = g.sort_values('n', ascending=False)

#     top = g_sorted.iloc[0]
#     top_sid, top_bin, top_offset = top['songID'], int(top['n']), int(top['stk'])
#     runner_bin = int(g_sorted.iloc[1]['n']) if len(g_sorted) > 1 else 0

#     per_song = g.groupby('songID')['n'].sum()
#     top_total = int(per_song.loc[top_sid])
#     unique_offsets = int(g[g['songID'] == top_sid]['stk'].nunique())

#     passes = True
#     reasons = []
#     if top_bin < MIN_TOP_BIN:
#         passes = False; reasons.append(f"top_bin<{MIN_TOP_BIN}")
#     if runner_bin and (top_bin / max(1, runner_bin) < MIN_RATIO):
#         passes = False; reasons.append(f"ratio<{MIN_RATIO}")
#     if top_total < MIN_TOTAL_MATCHES:
#         passes = False; reasons.append(f"total<{MIN_TOTAL_MATCHES}")
#     if unique_offsets < MIN_UNIQUE_OFFSETS:
#         passes = False; reasons.append(f"unique<{MIN_UNIQUE_OFFSETS}")

#     dbg = {
#         "top_sid": int(top_sid),
#         "top_bin": top_bin,
#         "runner_bin": runner_bin,
#         "top_total": top_total,
#         "unique_offsets": unique_offsets,
#         "n_fingerprints": n_fingerprints,
#         "passed": passes,
#         "fail_reasons": reasons
#     }
#     return (top_sid if passes else None), dbg

def pick_winner_with_confidence(df_matches, n_fingerprints, fingerprints=None):
    """
    Decide winner or None based on confidence gating.
    Now includes extra factors: coverage, offset consistency, Jaccard similarity.
    """
    if df_matches is None or df_matches.empty:
        return None, {"reason": "no matches"}

    if not isinstance(df_matches, pd.DataFrame):
        return None, {"reason": "bad df"}

    # Ensure df has songID and offset columns
    if 'songID' not in df_matches.columns:
        df_matches = df_matches.reset_index()
    if 'stk' not in df_matches.columns:
        df_matches = df_matches.reset_index()

    if 'songID' not in df_matches.columns or 'stk' not in df_matches.columns:
        return None, {"reason": "missing columns", "cols": list(df_matches.columns)}

    # Count matches per (songID, offset)
    g = df_matches.groupby(['songID', 'stk']).size().reset_index(name='n')
    g_sorted = g.sort_values('n', ascending=False)

    top = g_sorted.iloc[0]
    top_sid, top_bin, top_offset = top['songID'], int(top['n']), int(top['stk'])
    runner_bin = int(g_sorted.iloc[1]['n']) if len(g_sorted) > 1 else 0

    # Per-song totals
    per_song = g.groupby('songID')['n'].sum()
    top_total = int(per_song.loc[top_sid])

    # Distinct offsets (alignment bins) for the winning song
    unique_offsets = int(g[g['songID'] == top_sid]['stk'].nunique())

    # ---- NEW FACTORS ----
    # 1. Coverage = fraction of fingerprints explained by this song
    coverage = top_total / max(1, n_fingerprints)

    # 2. Offset consistency = how tightly aligned offsets are (stddev)
    offset_std = float(g[g['songID'] == top_sid]['stk'].std())
    if math.isnan(offset_std):
        offset_std = 0.0

    # 3. Jaccard similarity of hash sets
    jaccard = None
    if fingerprints is not None:
        chunk_hashes = {fp['hash'] for fp in fingerprints}
        db_hashes = {doc['hash'] for doc in db.find_fingerprints_by_song(top_sid)}
        denom = len(chunk_hashes | db_hashes)
        jaccard = len(chunk_hashes & db_hashes) / denom if denom > 0 else 0.0
        if math.isnan(jaccard):
            jaccard = 0.0
    # ---------------------
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

    # ---- Thresholds for new factors ----
    if coverage < 0.2:
        passes = False; reasons.append("low_coverage<0.2")
    if offset_std > 200:   # tweak threshold depending on scale of offsets
        passes = False; reasons.append("offset_std>200")
    if jaccard is not None and jaccard < 0.05:
        passes = False; reasons.append("jaccard<0.05")

    dbg = {
        "top_sid": int(top_sid),
        "top_bin": top_bin,
        "runner_bin": runner_bin,
        "top_total": top_total,
        "unique_offsets": unique_offsets,
        "coverage": coverage,
        "offset_std": offset_std,
        "jaccard": jaccard,
        "n_fingerprints": n_fingerprints,
        "passed": passes,
        "fail_reasons": reasons
    }

    return (top_sid if passes else None), dbg


def convert_to_mp3_8k(input_path: str, output_path: str):
    """Convert any audio file to mono MP3 at 8kHz."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(8000).set_channels(1)
    audio.export(output_path, format="mp3")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_in:
            tmp_in.write(await file.read())
            tmp_in.flush()
            tmp_input = tmp_in.name

        tmp_output = tmp_input + ".mp3"
        convert_to_mp3_8k(tmp_input, tmp_output)

        # Fingerprint
        data, rate = load_audio_data(tmp_output)
        fingerprints = search.get_fingerprints_from_audio(data, rate)

        if not fingerprints:
            return JSONResponse(content={"match": False, "reason": "no_fingerprints"})

        df_matches = search.get_df_of_fingerprint_offsets(fingerprints)
        if df_matches.empty:
            return JSONResponse(content={"match": False, "reason": "no_matches_in_db"})

        winner, dbg = pick_winner_with_confidence(df_matches, len(fingerprints))
        
        print(f"top_bin: {dbg['top_bin']}")
        print(f"unique_offsets: {dbg['unique_offsets']}")
        if dbg['top_bin'] >= 10:
            return JSONResponse(content={"match": True, "reason": "top_bin>=10", "metrics": dbg})
        if dbg['top_bin'] <= 10:
            return JSONResponse(content={"match": False, "reason": "top_bin<=10", "metrics": dbg})
        if winner is None:
            return JSONResponse(content={"match": False, "reason": "low_confidence", "metrics": dbg})

        song_doc = db.find_one_song({'_id': winner})
        return JSONResponse(content={
            "match": True,
            "honeypot": song_doc,
            "metrics": dbg
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(tmp_input)
            os.remove(tmp_output)
        except:
            pass

def process_file(file_path: str):
    """
    Process a single file: convert, fingerprint, and match against DB.
    """
    try:
        tmp_mp3 = file_path + ".mp3"
        convert_to_mp3_8k(file_path, tmp_mp3)

        data, rate = load_audio_data(tmp_mp3)
        fingerprints = search.get_fingerprints_from_audio(data, rate)
        if not fingerprints:
            return {"match": False, "reason": "no_fingerprints"}

        df_matches = search.get_df_of_fingerprint_offsets(fingerprints)
        if df_matches.empty:
            return {"match": False, "reason": "no_matches_in_db"}

        winner, dbg = pick_winner_with_confidence(df_matches, len(fingerprints))

        if dbg['top_bin'] >= 10:
            return JSONResponse(content={"match": True, "reason": "top_bin>=10", "metrics": dbg})
            print(True)
        if dbg['top_bin'] <= 10:
            return JSONResponse(content={"match": False, "reason": "top_bin<=10", "metrics": dbg})
            print(False)

        if winner is None:
            return {"match": False, "reason": "low_confidence", "metrics": dbg}

        song_doc = db.find_one_song({'_id': winner})
        return {"match": True, "honeypot": song_doc, "metrics": dbg}

    except Exception as e:
        return {"match": False, "error": str(e)}

    finally:
        try: os.remove(tmp_mp3)
        except: pass

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    # Expecting a .zip of audio files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        tmp_zip.write(await file.read())
        tmp_zip.flush()
        tmp_zip_path = tmp_zip.name

    results = []
    with tempfile.TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        for root, _, files in os.walk(extract_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    res = process_file(fpath)
                    res["file"] = fname
                    results.append(res)
                except Exception as e:
                    results.append({"file": fname, "error": str(e)})

    os.remove(tmp_zip_path)
    return JSONResponse(content={"results": results})

@app.post("/insert")
async def insert(file: UploadFile = File(...)):
    """
    Insert an audio file into the honeypot database.
    Converts to 8kHz mono mp3, fingerprints it, and saves in MongoDB.
    """
    try:
        # Save temp input
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_in:
            tmp_in.write(await file.read())
            tmp_in.flush()
            tmp_input = tmp_in.name

        tmp_mp3 = tmp_input + ".mp3"
        convert_to_mp3_8k(tmp_input, tmp_mp3)

        # Load audio + generate fingerprints
        data, rate = load_audio_data(tmp_mp3)
        fingerprints = search.get_fingerprints_from_audio(data, rate)

        if not fingerprints:
            return JSONResponse(content={"inserted": False, "reason": "no_fingerprints"})

        # Metadata for this song/document
        song_doc = {
            "artist": "Telecom Security",
            "album": "Honeypot DB",
            "title": os.path.splitext(file.filename)[0],
            "track_length_s": len(data) / rate
        }

        # Insert into DB
        song_id = db.insert_song(song_doc)
        db.insert_fingerprints(fingerprints, song_id)

        return JSONResponse(content={
            "inserted": True,
            "song_id": song_id,
            "song_doc": song_doc,
            "n_fingerprints": len(fingerprints)
        })

    except Exception as e:
        return JSONResponse(content={"inserted": False, "error": str(e)}, status_code=500)

    finally:
        try:
            os.remove(tmp_input)
            os.remove(tmp_mp3)
        except:
            pass