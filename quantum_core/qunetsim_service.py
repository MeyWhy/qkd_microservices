import threading
from qunetsim.components import Host, Network
from qunetsim.objects import Qubit


network: Network=None
alice: Host=None
bob: Host=None
lock =threading.Lock()
initialized =False


def initializenetwork()->None:
    global network, alice, bob, init

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
    init=True

    print("[QNetSim] Network ready. Alice <=> Bob connection established with successs")


def get_hosts():
    global init
    if not init:
        with lock:
            if not init:
                initializenetwork()
    return alice, bob


def encode_qubit(alice: Host, bit: int, basis: int) ->Qubit:
    q=Qubit(alice)

    if bit==1:
        q.X()  
    if basis==1:
        q.H()
    return q


def transmit_and_measure(bit: int, alice_basis: int, bob_basis: int) -> int:
    alice, bob=get_hosts()
    #encode & send step
    q=encode_qubit(alice, bit, alice_basis)
    alice.send_qubit("Bob", q, await_ack=True)

    #recv qubit by bob
    received_qubit=bob.get_data_qubit("Alice", wait=10)
    if received_qubit is None:
        raise RuntimeError("Bob did not receive the qubit (timeout).")

    #mesure in bob's basis
    if bob_basis==1:
        received_qubit.H()
    result=received_qubit.measure()
    return int(result)
