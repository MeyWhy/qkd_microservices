import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app=FastAPI(title="Alice service")

class AliceReq(BaseModel):
	size: int= Field(..., ge=1, le=10000)

def _get(url:str, label:str)-> dict:
	try:
		r=requests.get(url, timeout=5)
		r.raise_for_status()
		return r.json()
	except Exception as e:
		raise HTTPException(status_code=502, detail=f"{label} error: {e}")

@app.get("/health")
def health():
	return {"statut":"ok"}

@app.post("/alice")
def alice(req:AliceReq):
	bits=_get(f"http://localhost:8100/bits?size={req.size}", "bit_service")["bits"]
	basis=_get(f"http://localhost:8101/basis?size={req.size}", "basis_service")["basis"]
	return {"bits":bits, "basis":basis}

