"""
! pip install fastapi[all] python-jose[cryptography] passlib[bcrypt]
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional

# to be replaced with a database or more secure solution
fake_users_db = {
    "user1": {
        "username": "user1",
        "hashed_password": "$2b$12$K4J/...",
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# OAuth2 config
SECRET_KEY = "secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

"""
SET UP PASSWORD HASHING AND VERIFICATION
To ensure secure storage, we hash user passwords using passlib. 
A helper function verifies the password against the stored hash
"""

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

"""
CREATE USER AUTHENTICATION LOGIC
Functions are defined to look up a user in the database and authenticate them. 
The authenticate_user function verifies if a username-password pair is correct:
"""
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user['hashed_password']):
        return False
    return user

"""
GENERATE AN ACCESS TOKEN
To secure the endpoint, we’ll generate an access token that users need to access the protected route. 
This function creates a JWT token with an expiration time.
"""
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

"""
IMPLEMENT OAUTH2 DEPENDENCY TO EXTRACT USER FROM TOKEN
The function get_current_user will be used to verify the 
token’s validity each time the protected endpoint is accessed.
"""
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

"""
DEFINE THE LOGIN AND PROTECTED HELLO WORLD ENDPOINTS
Finally, we define two API endpoints. 
The /token endpoint allows users to request an access token, 
and the /hello endpoint, which requires the token, returns “Hello World” 
if the user is authenticated.
"""
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/hello")
async def read_hello_world(current_user: str = Depends(get_current_user)):
    return {"message": "Hello World"}
