import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.rag_service.api.fastapi_app:app", host="0.0.0.0", port=9000, reload=True)
