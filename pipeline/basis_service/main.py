import random
from fastapi import FastAPI, Query

app=FastAPI(title="Basis Service")
@app.get("/health")
def health():
    return {"statut": "ok"}

@app.get("/basis")
def generate_basis(n: int=Query(default=16, ge=1, le=1024)):
    basis=[]
    for _ in range(n):
        basis.append(random.randint(0,1))
    print(f"[basis_service] generated {n} bases --> {basis}")
    return{"basis": basis, "count": n}
