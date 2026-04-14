from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app=FastAPI(title="Key rate service")

class KeyRateReq(BaseModel):
    key_length: int=Field(..., ge=0, description="length of sifted key in bits")
    time_seconds: float=Field(..., gt=0, description="elapsed time in seconds")


@app.get("/health")
def health():
    return {"statut": "ok"}

@app.post("/keyrate")
def keyrate(req: KeyRateReq):
    rate=req.key_length/req.time_seconds
    return {"key_rate":round(rate, 6)}
