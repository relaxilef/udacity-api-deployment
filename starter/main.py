from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, world!"}

@app.post("/greet")
async def greet(request: Request):
    json_data = await request.json()
    name = json_data.get("name")
    return {"message": f"Hello, {name}"}
