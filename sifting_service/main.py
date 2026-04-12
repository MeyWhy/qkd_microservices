from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

app=FastAPI(title="sifting service")

class SiftReq(BaseModel):
	alice_bits:str
	alice_basis:str
	bob_basis:str
	bob_res:str

	@model_validator(mode="after")
	def check_length(self):
		lengths={"alice_bits":len(self.alice_bits),
			"alice_basis":len(self.alice_basis),
			"bob_basis":len(self.bob_basis),
			"bob_res":len(self.bob_res),
		}
		if len(set(lengths.values()))!=1:
			raise ValueError(f"All inputs must have the same length : {lengths}")
		if lengths["alice_bits"]==0:
			raise ValueError("inputs must not be empty")
		return self


@app.get("/health")
def health():
	return {"statut":"ok"}

@app.post("/sift")
def sift(req:SiftReq):
	sifted_key="".join(
	bob_bit for alice_basis, bob_basis, bob_bit in  zip(req.alice_basis, req.bob_basis, req.bob_res)
	if alice_basis== bob_basis
	)
	return {"sifted_key": sifted_key}

