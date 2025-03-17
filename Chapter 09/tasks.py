from fastapi import FastAPI, HTTPException
from typing import List, Optional

app = FastAPI()

# Keep track of todos. Does not persist if Python session is restarted.
_TODOS = {}

from pydantic import BaseModel
class TodoItem(BaseModel):
    todo: str

@app.post("/todos/{username}")
async def add_todo(username: str, todo_item: TodoItem):
    if username not in _TODOS:
        _TODOS[username] = []
    _TODOS[username].append(todo_item.todo)
    return {"status": "OK"}

@app.get("/todos/{username}", response_model=List[str])
async def get_todos(username: str):
    return _TODOS.get(username, [])

@app.delete("/todos/{username}")
async def delete_todo(username: str, todo_idx: int):
    if username not in _TODOS or not 0 <= todo_idx < len(_TODOS[username]):
        raise HTTPException(status_code=404, detail="Todo not found")
    _TODOS[username].pop(todo_idx)
    return {"status": "OK"}
