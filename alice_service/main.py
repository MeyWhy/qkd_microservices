import requests 
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, Field 
from qunetsim.components import Host, Network 
from qunetsim.objects import Qubit 

app=FastAPI(title="Alice service") 
class AliceReq(BaseModel): 
    size: int= Field(..., ge=1, le=500) 
    

def _get(url:str, label:str)-> dict: 
    try: 
        r=requests.get(url, timeout=5) 
        r.raise_for_status() 
        return r.json() 
    except Exception as e: 
        raise HTTPException(status_code=502, detail=f"{label} error: {e}") 

def _encode(host:Host, bit:str, basis:str)-> Qubit: 
    q=Qubit(host) 
    if bit=="1": 
        q.X() 
    if basis=="X": 
        q.H() 
    return q 

def _transmit(bits:str, basis:str)-> None: 
    network=Network.get_instance() 
    alice=Host("Alice") 
    bob=Host("Bob") 
    alice.add_connection("Bob") 
    bob.add_connection("Alice") 
    network.add_hosts([alice,bob]) 
    network.start() 
    alice.start() 
    bob.start() 
    try: 
        for bit, base in zip(bits,basis): 
            q=_encode(alice, bit, base) 
            alice.send_qubit("Bob", q, await_ack=True) 
    finally: 
        network.stop(stop_hosts=True)      

@app.get("/health") 
def health(): 
    return {"statut":"ok"} 

@app.post("/alice") 
def alice(req:AliceReq):         
    bits=_get(f"http://localhost:8100/bits?size={req.size}", "bit_service")["bits"] 
    basis=_get(f"http://localhost:8101/basis?size={req.size}", "basis_service")["basis"] 
    _transmit(bits,basis) 
    
    return {"bits":bits, "basis":basis}
