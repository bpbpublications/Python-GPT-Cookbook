from fastapi import FastAPI, HTTPException, Header

app = FastAPI()

API_KEY = "your_secret_api_key"  # Replace with your actual API key

@app.get("/")
async def read_root(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"message": "Hello World"}
