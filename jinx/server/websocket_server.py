import hashlib
import os
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sympy.physics.vector.printing import params

from rag.config import DATA_PATH
from rag.rag_system import HarryPotterRAG  # NEW
from server.templates import PROMPT_TEMPLATES

app = FastAPI()
rag = HarryPotterRAG()


@app.on_event("startup")
async def startup_event():
    current_hash = compute_documents_hash(DATA_PATH)
    stored_hash = read_stored_hash()

    if stored_hash != current_hash:
        print("🆕 Documents changed or vectorstore missing — rebuilding...")
        rag.build_index_pipeline()
        write_hash(current_hash)
    else:
        try:
            rag.load_vectorstore()
            print("✅ Vectorstore loaded")
        except ValueError:
            print("❌ Vectorstore missing but hash unchanged — rebuilding just in case...")
            rag.build_index_pipeline()
            write_hash(current_hash)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    params = websocket.query_params
    try:
        while True:
            data = await websocket.receive_text()
            voice = params.get("voice", "en-US-Wavenet-D")
            print(f"💬 Received: {data} {voice}")

            if voice not in PROMPT_TEMPLATES:
                await websocket.send_text(f"❌ Error: Voice '{voice}' not supported.")
                await websocket.send_text("__END__")
                continue  # Skip processing this message

            generator = rag.query(query_text=data, style=voice)
            async for chunk in stream_chunks(generator):
                await websocket.send_text(chunk)
            await websocket.send_text("__END__")

    except WebSocketDisconnect:
        print("🔌 Client disconnected")


    except WebSocketDisconnect:
        print("🔌 Client disconnected")

@app.get("/prompt-styles", response_model=List[dict])
def get_prompt_styles():
    return [
        {"key": key.lower().replace(" ", "_"), "display_name": key}
        for key in PROMPT_TEMPLATES.keys()
    ]


async def stream_chunks(generator):
    for chunk in generator:
        if isinstance(chunk, dict):
            continue  # You can expand this to send source docs
        yield chunk


def compute_documents_hash(data_path: str = DATA_PATH) -> str:
    hash_md5 = hashlib.md5()

    for root, _, files in sorted(os.walk(data_path)):
        for fname in sorted(files):
            if fname.endswith(".md"):
                path = os.path.join(root, fname)
                with open(path, "rb") as f:
                    while chunk := f.read(4096):
                        hash_md5.update(chunk)

    return hash_md5.hexdigest()


def read_stored_hash(filepath="vectorstore_hash.txt") -> str:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
    return ""


def write_hash(hash_value: str, filepath="vectorstore_hash.txt"):
    with open(filepath, "w") as f:
        f.write(hash_value)
