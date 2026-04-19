from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app=FastAPI(title="Sifting Service")

class SiftReq(BaseModel):
    alice_bits:List[int]
    alice_basis: List[int]
    bob_basis: List[int]
    bob_res:List[Optional[int]]

@app.get("/health")
def health():
    return {"statut":"ok"}

@app.post("/sift")
def sift(req:SiftReq):
    n=len(req.alice_bits)

    alice_sifted=[]
    bob_sifted=[]
    for i in range(n):
        if req.alice_basis[i]==req.bob_basis[i] and req.bob_res[i] is not None:
            alice_sifted.append(req.alice_bits[i])
            bob_sifted.append(req.bob_res[i])

    print(f"[siftinger_service] {len(alice_sifted)}/{n} bits kept after the sifting phase")
    return {
        "alice_sifted":alice_sifted,
        "bob_sifted":bob_sifted,
        "sifted_length":len(alice_sifted),
    }
