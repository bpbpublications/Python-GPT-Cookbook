from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader

app = FastAPI()

# OAuth (GitHub) setup and dummy validation function
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def validate_oauth_token(token: str = Depends(oauth2_scheme)):
    # Replace with actual OAuth validation logic
    if token != "valid_oauth_token":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

# Service level auth setup and dummy validation function
api_key_header = APIKeyHeader(name="X-API-Key")

def validate_api_key(api_key: str = Depends(api_key_header)):
    # Replace with actual API key validation logic
    if api_key != "valid_api_key":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

# Create the Endpoints
@app.get("/noauth")
async def no_auth_hello():
    return {"message": "Hello World with No Auth"}

@app.get("/oauth")
async def oauth_hello(token: str = Depends(validate_oauth_token)):
    return {"message": "Hello World with OAuth"}

@app.get("/serviceauth")
async def service_auth_hello(api_key: str = Depends(validate_api_key)):
    return {"message": "Hello World with Service Level Auth"}
