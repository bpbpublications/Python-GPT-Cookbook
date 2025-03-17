from fastapi import FastAPI

app = FastAPI()

@app.get("/nonconsequential")
async def non_consequential_hello():
    return {"message": "Hello World from a non-consequential action"}

@app.post("/consequential")
async def consequential_hello():
    return {"message": "Hello World from a consequential action"}
