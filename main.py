import uvicorn

if __name__ == "__main__":
    # Use the import string format "src.main:app" so the reload feature works
    uvicorn.run(
        "src.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )