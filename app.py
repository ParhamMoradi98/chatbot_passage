from flask import Flask, request, jsonify, session 
from flask_cors import CORS
from flask_session import Session

import sqlite3
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import sys
import requests  

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = "my_secret_key_here"  
Session(app)

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not set in environment variables.")

DEEPSEEK_CHAT_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

def load_system_data():
    try:
        with open("system_state.txt", "r") as f:
            data = json.load(f)
            data.setdefault("state", "video_selection")
            data.setdefault("candidate_list", [])
            data.setdefault("selected_video", None)
            return data
    except Exception as e:
        return {"state": "video_selection", "candidate_list": [], "selected_video": None}

def update_system_data(state, candidate_list=None, selected_video=None):
    data = {
        "state": state,
        "candidate_list": candidate_list or [],
        "selected_video": selected_video
    }
    with open("system_state.txt", "w") as f:
        json.dump(data, f)
    print("DEBUG: System data updated:", data, file=sys.stderr)



def load_caption_embeddings(batch_size=1000):
    """
    Generator yielding batched rows of (id, video_id, chunk_order, text, embedding)
    from the captions table.
    """
    conn = sqlite3.connect('videos.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, video_id, chunk_order, text, embedding
        FROM captions
        WHERE embedding IS NOT NULL
    """)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield row
    conn.close()

conn = sqlite3.connect('videos.db', timeout=30)
cursor = conn.cursor()
cursor.execute("SELECT embedding FROM captions WHERE embedding IS NOT NULL LIMIT 1")
first_row = cursor.fetchone()
conn.close()

if not first_row:
    raise ValueError("No embeddings found in the 'captions' table. Ensure this table is populated with caption embeddings.")

first_embedding = np.array(json.loads(first_row[0])).astype('float32')
dimension = first_embedding.shape[0]

index = faiss.IndexFlatL2(dimension)
metadata_list = []
batch_embeddings = []
batch_metadata = []
batch_size = 10000
total_indexed = 0

print("Loading caption embeddings and building FAISS index for captions...")
for row in load_caption_embeddings(batch_size):
    cap_id, video_id, chunk_order, text, embedding_json = row
    embedding_vector = np.array(json.loads(embedding_json)).astype('float32')
    batch_embeddings.append(embedding_vector)
    batch_metadata.append({
        'id': cap_id,
        'video_id': video_id,
        'chunk_order': chunk_order,
        'text': text
    })
    if len(batch_embeddings) >= batch_size:
        batch_embeddings_np = np.vstack(batch_embeddings)
        index.add(batch_embeddings_np)
        metadata_list.extend(batch_metadata)
        total_indexed += len(batch_embeddings)
        print(f"Indexed {total_indexed} caption embeddings so far.")
        batch_embeddings = []
        batch_metadata = []
if batch_embeddings:
    batch_embeddings_np = np.vstack(batch_embeddings)
    index.add(batch_embeddings_np)
    metadata_list.extend(batch_metadata)
    total_indexed += len(batch_embeddings)

print(f"Total caption embeddings indexed: {total_indexed}")
print(f"Final FAISS caption index size: {index.ntotal}")



def load_description_embeddings():
    """
    Load all (id, video_id, description, description_embedding) rows
    from the videos table where description_embedding is not null.
    """
    conn = sqlite3.connect('videos.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, video_id, description, description_embedding
        FROM videos
        WHERE description_embedding IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

conn = sqlite3.connect('videos.db', timeout=30)
cursor = conn.cursor()
cursor.execute("SELECT description_embedding FROM videos WHERE description_embedding IS NOT NULL LIMIT 1")
first_desc_row = cursor.fetchone()
conn.close()

desc_index = None
desc_metadata_list = []
if first_desc_row is not None:
    print("Loading description embeddings and building FAISS index for descriptions...")
    first_desc_embedding = np.array(json.loads(first_desc_row[0])).astype('float32')
    desc_dimension = first_desc_embedding.shape[0]
    desc_index = faiss.IndexFlatL2(desc_dimension)

    desc_rows = load_description_embeddings()
    desc_embeddings = []
    for row in desc_rows:
        db_id, video_id, description, desc_emb_json = row
        emb_vector = np.array(json.loads(desc_emb_json)).astype('float32')
        desc_index.add(np.expand_dims(emb_vector, axis=0))
        desc_metadata_list.append({
            'id': db_id,
            'video_id': video_id,
            'description': description
        })
    print(f"Total description embeddings indexed: {desc_index.ntotal}")
else:
    print("No description embeddings found in the 'videos' table.")

model = SentenceTransformer('all-MiniLM-L6-v2')

def query_captions(query_text, k=30):
    """
    Perform a FAISS search on the captions index.
    Returns top-k results with their respective distance.
    """
    query_embedding = model.encode(query_text).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            'metadata': metadata_list[idx],
            'distance': distances[0][i]
        })
    return results

def query_descriptions(query_text, k=30):
    """
    Perform a FAISS search on the descriptions index.
    Returns top-k results with their respective distance.
    """
    if desc_index is None or desc_index.ntotal == 0:
        return []

    query_embedding = model.encode(query_text).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = desc_index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            'metadata': desc_metadata_list[idx],
            'distance': distances[0][i]
        })
    return results

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    return None

def get_video_info(video_id):
    conn = sqlite3.connect('videos.db')
    cursor = conn.cursor()
    cursor.execute("SELECT url, description FROM videos WHERE video_id = ?", (video_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return f"Video {video_id}", "No description available."


def search_video_by_description(query_text):
    """
    Searches the 'videos' table for a video whose description (in lowercase)
    contains the query_text (also in lowercase). Returns the video_id if found.
    """
    conn = sqlite3.connect('videos.db')
    cursor = conn.cursor()
    sql = "SELECT video_id FROM videos WHERE LOWER(description) LIKE ? LIMIT 1;"
    cursor.execute(sql, ('%' + query_text.lower() + '%',))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get('question', '').strip()
    if not user_input:
        return jsonify({'answer': "Empty input received. Please enter your question."})
    
    if user_input.lower() == "exit":
        session.clear()
        update_system_data("video_selection", candidate_list=[], selected_video=None)
        session['state'] = "video_selection"
        return jsonify({'answer': "System State: Video Selection\n\nExiting current context. Please enter a YouTube URL or search text."})
    
    sys_data = load_system_data()
    session['state'] = sys_data.get("state", "video_selection")
    session['candidate_list'] = sys_data.get("candidate_list", [])
    if "selected_video" in sys_data:
        session['selected_video'] = sys_data["selected_video"]
    current_state = session['state']
    print("DEBUG: Loaded state from file:", current_state, file=sys.stderr)
    
    def format_state_message(msg, state_label):
        return f"System State: {state_label.replace('_', ' ').title()}\n\n{msg}"
    
    # --- State: Video Selection ---
    if current_state == "video_selection":
        if "youtube.com" in user_input or "youtu.be" in user_input:
            video_id = extract_video_id(user_input)
            if not video_id:
                return jsonify({'answer': format_state_message("Could not extract video ID from URL. Please try again.", "video selection")})
            session['selected_video'] = video_id
            update_system_data("ask_about_video", candidate_list=[], selected_video=video_id)
            session['state'] = "ask_about_video"
            return jsonify({'answer': format_state_message(f"Video {video_id} selected. Now ask your question about this video.", "ask about video")})
        else:
            # First, try to find an exact match from the video descriptions.
            exact_video = search_video_by_description(user_input)
            candidate_list = []
            if exact_video:
                candidate_list.append(exact_video)
            # Then, get candidates via the embedding-based query.
            results = query_captions(user_input, k=30)
            for res in results:
                vid = res['metadata']['video_id']
                if vid not in candidate_list:
                    candidate_list.append(vid)
                if len(candidate_list) == 5:
                    break
            if not candidate_list:
                return jsonify({'answer': format_state_message("No matching videos found. Please try another query.", "video selection")})
            session['candidate_list'] = candidate_list
            update_system_data("video_selection_confirmation", candidate_list=candidate_list, selected_video=session.get('selected_video'))
            session['state'] = "video_selection_confirmation"
            message = "Select a video by entering the corresponding number:\n\n"
            for i, vid in enumerate(candidate_list, start=1):
                url, description = get_video_info(vid)
                message += (f"{i}:\nTitle: {vid}\nURL: {url}\nDescription: {description}\n\n")
            return jsonify({'answer': format_state_message(message, "video selection confirmation")})
    
    # --- State: Video Selection Confirmation ---
    elif current_state == "video_selection_confirmation":
        candidate_list = session.get('candidate_list', [])
        print("DEBUG: Candidate list from session:", candidate_list, file=sys.stderr)
        if not candidate_list:
            update_system_data("video_selection", candidate_list=[], selected_video=None)
            session['state'] = "video_selection"
            return jsonify({'answer': format_state_message("No candidate list available. Please enter a YouTube URL or search text.", "video selection")})
        if user_input.isdigit():
            selection_num = int(user_input)
            if 1 <= selection_num <= len(candidate_list):
                selected_video = candidate_list[selection_num - 1]
                session['selected_video'] = selected_video
                session.pop('candidate_list', None)
                update_system_data("ask_about_video", candidate_list=[], selected_video=selected_video)
                session['state'] = "ask_about_video"
                return jsonify({'answer': format_state_message(f"Video {selected_video} selected. Now ask your question about this video.", "ask about video")})
            else:
                return jsonify({'answer': format_state_message("Invalid selection number. Please try again.", "video selection confirmation")})
        else:
            message = "Please select a video by entering a valid number from the candidate list:\n\n"
            for i, vid in enumerate(candidate_list, start=1):
                url, description = get_video_info(vid)
                message += (f"{i}:\nTitle: {vid}\nURL: {url}\nDescription: {description}\n\n")
            return jsonify({'answer': format_state_message(message, "video selection confirmation")})
    
    # --- State: Ask About Video ---
    elif current_state == "ask_about_video":
        selected_video = session.get('selected_video')
        if not selected_video:
            update_system_data("video_selection", candidate_list=[], selected_video=None)
            session['state'] = "video_selection"
            return jsonify({'answer': format_state_message("No video selected. Please enter a YouTube URL or search text.", "video selection")})
        results = query_captions(user_input, k=30)
        filtered_results = [res for res in results if res['metadata']['video_id'] == selected_video]
        if not filtered_results:
            return jsonify({'answer': format_state_message("No relevant information found for this video. Try asking a different question.", "ask about video")})
        top_chunks = filtered_results[:15]
        chunks_text = "\n".join([f"- {chunk['metadata']['text']}" for chunk in top_chunks])
        prompt = f" This is the Context:\n{chunks_text}\n\n Question:\n{user_input}\n\nAnswer short and be explicit:"
        print("DEBUG: Prompt for Deepseek LLM:", prompt, file=sys.stderr)
    

        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 512
            }
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            deepseek_response = requests.post(
                DEEPSEEK_CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if deepseek_response.status_code == 200:
                deepseek_data = deepseek_response.json()
                answer_text = deepseek_data.get("choices", [{}])[0].get("message", {}).get("content", "No answer provided by Deepseek LLM.")
            else:
                answer_text = f"Deepseek LLM API error: {deepseek_response.status_code} - {deepseek_response.text}"
                
        except Exception as e:
            answer_text = f"Error contacting Deepseek LLM API: {str(e)}"
            
        return jsonify({'answer': format_state_message(answer_text, "ask about video")})
    

    else:
        update_system_data("video_selection", candidate_list=[], selected_video=None)
        session['state'] = "video_selection"
        return jsonify({'answer': format_state_message("Resetting conversation. Please enter a YouTube URL or search text.", "video selection")})

if __name__ == '__main__':
    app.run()
