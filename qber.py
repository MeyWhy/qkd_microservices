from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app=FastAPI(title="Qber service")

class QBERReq(BaseModel):
    alice_sifted:List[int]
    bob_sifted: List[int]
    sample_fraction: float=0.5
    #fraction of sifted bits used to estimate qber


@app.get("/health")
def health():
    return {"statut": "ok"}

@app.post("/qber")
def compute_qber(req:QBERReq):
    n=len(req.alice_sifted)
    if n==0:
        return{"qber":None, "error":"Empty sifted key"}

    sample_size=max(1,int(n*req.sample_fraction))
    errors=0
    for a,b in zip(req.alice_sifted[:sample_size], req.bob_sifted[:sample_size]):
        if a!=b:
            errors+=1
    qber=errors/sample_size
    print(f"[qber_service] sample_size={sample_size}, errors={errors}, qber={qber:.4f}")
    return {
        "qber":qber,
        "errors":errors,
        "sample_size":sample_size,
        "total_sifted":n,
    }
