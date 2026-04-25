from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid

class Basis(str, Enum):
    RECTILINEAR="Z"
    DIAGONAL="X"

def new_session_id() ->str:
    return str(uuid.uuid4())

class NetworkInitReq(BaseModel):
    session_id:str
    n_qubits:int= Field(gt=0, le=10000)

class NetworkInitResp(BaseModel):
    session_id: str
    statut:str
    message: str=""

class SendQubitReq(BaseModel):
    session_id:str
    qubit_id: int
    bit: int=Field(ge=0, le=1)
    basis:Basis

class SendQubitResp(BaseModel):
    qubit_id:int
    delivered:bool

class GetMeasurementsReq(BaseModel):
    session_id:str

class QubitMeasurement(BaseModel):
    qubit_id:int
    basis: Basis
    bit_res:int=Field(ge=0, le=1)

class GetMeasurementsResp(BaseModel):
    session_id:str
    measurements: list[QubitMeasurement]

class NetworkStopReq(BaseModel):
    session_id:str

class SessionStartReq(BaseModel):
    n_qubits:int=Field(default=200, gt=0, le=5000)
    loss_rate:float=Field(default=0.0, gt=0.0, le=1.0)

class SessionStartResp(BaseModel):
    session_id: str
    statut: str
    n_qubits_sent: int
    n_qubits_received: int
    n_sifted: int
    qber: float
    key_final:str #TODO to be changed to its hash for more secu
    error_message: str=""

class SiftReq(BaseModel):
    session_id:str
    alice_bases:list[tuple[int, Basis]]

class SiftResp(BaseModel):
    session_id:str
    bob_bases: list[tuple[int, Basis]]
    n_sifted:int

class BobSessionState(BaseModel):
    session_id:str
    measurements: list[QubitMeasurement]= []
    sifted_bits: list[int]=[]
    statut: str="measuring" #sifted, or done


class BB84Error(BaseModel):
    code: str
    message:  str
    session_id: Optional[str]= None

class ErrorCode:
    SESSION_NOT_FOUND="SESSION_NOT_FOUND"
    NETWORK_UNAVAILABLE="NETWORK_UNAVAILABLE"
    QBER_TOO_HIGH="QBER_TOO_HIGH"
    INSUFFICIENT_BITS="INSUFFICIENT_BITS"
    TIMEOUT="TIMEOUT"
    INTERNAL_ERROR="INTERNAL_ERROR"