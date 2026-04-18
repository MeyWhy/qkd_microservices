from quantum_core.celery_app import celery_app
from quantum_core.qunetsim_service import transmit_and_measure


@celery_app.task(name="quantum_core.tasks.transmit_qubit", bind=True)
def transmit_qubit(self, bit: int, alice_basis: int, bob_basis:int, index: int=0)-> int:

    print(
        f"[Task] qubit[{index}]: bit={bit} alice_basis={alice_basis} bob_basis={bob_basis}"
    )
    result=transmit_and_measure(
        bit=bit,
        alice_basis=alice_basis,
        bob_basis=bob_basis,
    )
    print(f"[task] qubit[{index}]: Bob measured ==> {result}")
    return result
