from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

app = FastAPI(title="QBER Service")


class QBERReq(BaseModel):
    alice_sifted: str
    bob_sifted: str

    @model_validator(mode="after")
    def check(self):
        if not self.alice_sifted or not self.bob_sifted:
            raise ValueError("Inputs must not be empty")

        if len(self.alice_sifted) != len(self.bob_sifted):
            raise ValueError("alice_sifted and bob_sifted must have same length")

        return self


@app.get("/health")
def health():
    return {"statut": "ok"}


@app.post("/qber")
def qber(req: QBERReq):
    if len(req.alice_sifted) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty sifted key → cannot compute QBER"
        )

    errors = sum(
        1 for a, b in zip(req.alice_sifted, req.bob_sifted) if a != b
    )

    qber_value = errors / len(req.alice_sifted)

    return {"qber": round(qber_value, 6)}
