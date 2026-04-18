from fastapi import FastAPI
from pydantic import BaseModel
import math

app=FastAPI(title="Keyrate service")

class KeyrateReq(BaseModel):
    total_bits:int
    sifted_length:int
    qber:float
    sample_fraction: float=0.5
    #fraction used for qber estimation

@app.get("/health")
def health():
    return {"statut": "ok"}

def h(p: float)->float:
        if p<=0 or p>=1:
            return 0.0
        return -p*math.log2(p) - (1-p)*math.log2(1-p)
@app.post("/keyrate")
def compute_key_rate(req: KeyrateReq):
    sifted_remaining=int(req.sifted_length*(1-req.sample_fraction))
    secret_fraction=1-2*h(req.qber)
    if secret_fraction<0:
         secret_fraction=0.0
    final_key_bits=int(sifted_remaining*secret_fraction)
    if req.total_bits>0:
        key_rate=final_key_bits/req.total_bits
    else:
        key_rate=0.0

    print(f"[keyrate_service]sifted_remaining={sifted_remaining}, "f"secret_fraction={secret_fraction:.4f}, "f"final_key_bits={final_key_bits}, key_rate={key_rate:.4f}")
    return{"final_key_bits":final_key_bits,"key_rate": key_rate,"sifted_remaining": sifted_remaining,"secret_fraction": secret_fraction,}
