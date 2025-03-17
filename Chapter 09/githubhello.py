from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from datetime import datetime, timedelta
import httpx

app = FastAPI()

# GitHub OAuth configuration
CLIENT_ID = "your_github_client_id"
CLIENT_SECRET = "your_github_client_secret"
REDIRECT_URI = "your_localtunnel_redirect_uri"  # Update with LocalTunnel URI
AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Function to create JWT token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

"""
SET UP THE REDIRECT TO GITHUB’S AUTHORIZATION PAGE
Next, we create an endpoint that redirects users to GitHub’s authorization page, 
where they’ll log in and authorize access to your application:
"""
@app.get("/auth")
async def auth(request: Request):
    return RedirectResponse(
        url=f"{AUTHORIZE_URL}?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    )

"""
HANDLE THE CALLBACK AND RETRIEVE THE GITHUB ACCESS TOKEN
Once the user authorizes the application, GitHub sends a code back to our callback endpoint. 
This code allows us to request an access token from GitHub, 
which we use to generate a JWT for application access:
"""
@app.get("/callback")
async def callback(code: str):
    token_response = httpx.post(
        TOKEN_URL,
        headers={"Accept": "application/json"},
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
    )
    token_json = token_response.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Invalid GitHub token")

    jwt_token = create_access_token(
        data={"sub": access_token}, 
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": jwt_token, "token_type": "bearer"}

"""
DEFINE THE PROTECTED “HELLO WORLD” ENDPOINT
Finally, we create a simple “Hello World” endpoint that is 
protected by JWT authentication. Only users with a valid 
JWT can access this endpoint:
"""
@app.get("/hello")
async def read_hello_world(token: str = Depends(oauth2_scheme)):
    return {"message": "Hello World"}
