import threading
from qunetsim.components import Host, Network
from qunetsim.objects import Qubit


network: Network=None
alice: Host=None
bob: Host=None
lock =threading.Lock()
ready =False

qubit_lock=threading.Lock()

def initializenetwork()->None:
    global network, alice, bob, ready

    print("[QuNetSim] initializing quantum network...")

    network=Network.get_instance()
    network.start()
    alice=Host("Alice")
    bob=Host("Bob")
    #establishing a quantum+classic channel between a & b
    alice.add_connection("Bob")
    bob.add_connection("Alice")
    alice.start()
    bob.start()
    network.add_host(alice)
    network.add_host(bob)
    ready=True

    print("[QNetSim] Network ready. Alice <=> Bob connection established with successs")


def get_hosts():
    global ready
    if not ready:
        with lock:
            if not ready:
                initializenetwork()
    return alice, bob


def encode(host, bit: int, basis: int):
    q=Qubit(host)
    if bit==1:#flip state
        q.X()  
    if basis==1:
        q.H()#change basis
    return q


def measure(qubit, basis: int)->int:
    if basis==1:
        qubit.H()
    result=qubit.measure()
    return int(result)

def transmit_qubit_event(event: dict)->int:
    bit=int(event["bit"])
    alice_basis=int(event["alice_basis"])
    bob_basis=int(event["bob_basis"])
    index=event.get("index", "?")
    session_id=event.get("session_id", "?")
    alice,bob=get_hosts()

    #1 qubit a la fois !!!
    with qubit_lock:
        q=encode(alice, bit, alice_basis)
        alice.send_qubit("Bob", q, await_ack=False)

        received=bob.get_data_qubit("Alice", wait=5)
        if received is None:
            raise RuntimeError(f"[quantum_core] qubit[{index}] session={session_id}:""Bob did not receive qubit(timeout)")
        
        result=measure(received, bob_basis)
        print(f"{ session_id}:  {index} | {bit} | {alice_basis} | {bob_basis}, {result}")
        return result
