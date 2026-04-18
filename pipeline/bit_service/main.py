import random
from fastapi import FastAPI, Query

app=FastAPI(title="Bit Service")

@app.get("/health")
def health():
    return {"statut": "ok"}

@app.get("/bits")
def generate_bits(n: int=Query(default=16, ge=1, le=1024)):
    bits=[]
    for _ in range(n):
        bits.append(random.randint(0,1))
    print(f"[bit_service] generated {n} bits --> {bits}")
    return {"bits": bits,"count": n}
