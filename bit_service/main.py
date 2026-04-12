from fastapi import FastAPI, Query, HTTPException
import secrets

app = FastAPI(title="Bit generator")


@app.get("/health")
def health():
    return {"statut": "ok"}


@app.get("/bits")
def get_bits(size:int=Query(..., ge=1, le=10000, description="Number of bits to generate")):
	bits="".join(str(secrets.randbits(1))for _ in range(size))
	return {"bits":bits}


