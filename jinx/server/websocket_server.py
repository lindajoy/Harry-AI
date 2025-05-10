from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rag.rag_system import HarryPotterRAG  # NEW

app = FastAPI()
rag = HarryPotterRAG()

@app.on_event("startup")
async def startup_event():
    try:
        rag.load_vectorstore()
        print("✅ Vectorstore loaded")
    except ValueError:
        print("🔄 Vectorstore not found, building...")
        rag.build_index_pipeline()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"💬 Received: {data}")

            generator = rag.query(data)
            async for chunk in stream_chunks(generator):
                await websocket.send_text(chunk)
            await websocket.send_text("__END__")

    except WebSocketDisconnect:
        print("🔌 Client disconnected")

async def stream_chunks(generator):
    for chunk in generator:
        if isinstance(chunk, dict):
            continue  # You can expand this to send source docs
        yield chunk
