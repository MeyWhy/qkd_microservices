from fastapi import FastAPI
from pydantic import BaseModel, model_validator

app = FastAPI(title="Sifting service")


class SiftReq(BaseModel):
    alice_bits: str
    alice_basis: str
    bob_basis: str
    bob_res: str

    @model_validator(mode="after")
    def check_length(self):
        lengths = {
            "alice_bits": len(self.alice_bits),
            "alice_basis": len(self.alice_basis),
            "bob_basis": len(self.bob_basis),
            "bob_res": len(self.bob_res),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(f"All inputs must have the same length: {lengths}")

        if lengths["alice_bits"] == 0:
            raise ValueError("inputs must not be empty")

        return self


@app.get("/health")
def health():
    return {"statut": "ok"}


@app.post("/sift")
def sift(req: SiftReq):
    alice_sifted = []
    bob_sifted = []

    for a_bit, a_basis, b_basis, b_bit in zip(
        req.alice_bits,
        req.alice_basis,
        req.bob_basis,
        req.bob_res,
    ):
        if a_basis == b_basis:
            alice_sifted.append(a_bit)
            bob_sifted.append(b_bit)

    alice_sifted = "".join(alice_sifted)
    bob_sifted = "".join(bob_sifted)

    return {
        "alice_sifted": alice_sifted,
        "bob_sifted": bob_sifted,
        "sifted_key": bob_sifted,
    }
