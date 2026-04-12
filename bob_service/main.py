import requests
import secrets
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

app=FastAPI(title="Bob service")

class BobReq(BaseModel):
	alice_bits:str
	alice_basis:str
	size:int= Field(..., ge=1, le=10000)

	@model_validator(mode="after")
	def check_consistency(self):
		if len(self.alice_bits)!=self.size or len(self.alice_basis)!=self.size:
			raise ValueError(f"alice_bits and alice_basis must have the same length as size ({self.size}")
		return self


def _get_basis(size:int)-> str:
        try:
                r=requests.get(f"http://localhost:8101/basis?size={size}", timeout=5)
                r.raise_for_status()
                return r.json()["basis"]
        except Exception as e:
                raise HTTPException(status_code=502, detail=f"basis_service error: {e}")

@app.get("/health")
def health():
        return {"statut":"ok"}

@app.post("/bob")
def bob(req:BobReq):
	bob_basis=_get_basis(req.size)
	bob_res="".join(
	alice_bit if bob_basis==alice_basis else str(secrets.randbits(1))
	for alice_bit, alice_basis, bob_basis in zip(req.alice_bits, req.alice_basis, bob_basis)
	)
	return {"bob_basis":bob_basis, "bob_res": bob_res}
