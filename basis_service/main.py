from fastapi import FastAPI, Query
import secrets

app=FastAPI(title="Basis Generator")

@app.get("/health")
def health():
	return {"statut":"ok"}

@app.get("/basis")
def get_basis(size: int=Query(..., ge=1, le=10000)):
	basis="".join(secrets.choice("ZX")for _ in range(size))
	return {"basis":basis}
