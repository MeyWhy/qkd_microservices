from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid
import time

class SessionStatus(str, Enum):
    CREATED      = "created"       # Session créée, pas encore démarrée
    INITIALIZING = "initializing"  # QNS + Bob en cours d'init
    SENDING      = "sending"       # Qubits en transit (workers Celery)
    SIFTING      = "sifting"       # Chord terminé, sifting en cours
    DONE         = "done"          # Clé générée avec succès
    ABORTED      = "aborted"       # Erreur non récupérable


# Transitions valides : état_courant → {états_suivants autorisés}
VALID_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.CREATED:      {SessionStatus.INITIALIZING, SessionStatus.ABORTED},
    SessionStatus.INITIALIZING: {SessionStatus.SENDING,      SessionStatus.ABORTED},
    SessionStatus.SENDING:      {SessionStatus.SIFTING,      SessionStatus.ABORTED},
    SessionStatus.SIFTING:      {SessionStatus.DONE,         SessionStatus.ABORTED},
    SessionStatus.DONE:         set(),   # état terminal
    SessionStatus.ABORTED:      set(),   # état terminal
}

# États terminaux — aucune transition possible
TERMINAL_STATES = {SessionStatus.DONE, SessionStatus.ABORTED}


class TransitionError(Exception):
    pass


def validate_transition(current: SessionStatus, target: SessionStatus) -> None:

    if target not in VALID_TRANSITIONS.get(current, set()):
        raise TransitionError(
            f"Transition interdite : {current.value} → {target.value}. "
            f"Transitions valides : {[s.value for s in VALID_TRANSITIONS[current]]}"
        )

class OrchestratorSession(BaseModel):

    session_id:   str
    status:       SessionStatus = SessionStatus.CREATED
    n_qubits:     int
    batch_size:   int
    loss_rate:    float

    # Timestamps pour métriques de latence
    created_at:      float = Field(default_factory=time.time)
    started_at:      Optional[float] = None
    sending_at:      Optional[float] = None
    sifting_at:      Optional[float] = None
    completed_at:    Optional[float] = None

    # Résultats intermédiaires
    celery_task_id:  Optional[str] = None   # ID du chord Celery
    n_delivered:     int   = 0
    n_sifted:        int   = 0
    qber:            float = 0.0
    key_final:        str   = ""
    error_message:   str   = ""

    def transition(self, target: SessionStatus) -> "OrchestratorSession":
        """
        Applique une transition et horodate.
        Retourne self pour le chaînage.
        """
        validate_transition(self.status, target)
        self.status = target

        now = time.time()
        if target == SessionStatus.INITIALIZING:
            self.started_at  = now
        elif target == SessionStatus.SENDING:
            self.sending_at  = now
        elif target == SessionStatus.SIFTING:
            self.sifting_at  = now
        elif target in TERMINAL_STATES:
            self.completed_at = now

        return self

    @property
    def elapsed_s(self) -> float:
        if self.completed_at and self.created_at:
            return self.completed_at - self.created_at
        return time.time() - self.created_at

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES

class SessionStartRequest(BaseModel):
    n_qubits:   int   = Field(default=200, gt=0, le=5_000)
    loss_rate:  float = Field(default=0.0,  ge=0.0, le=1.0)
    batch_size: int   = Field(default=10,   gt=0,   le=100)


class SessionStatusResponse(BaseModel):
    session_id:    str
    status:        SessionStatus
    elapsed_s:     float
    n_qubits:      int
    n_delivered:   int   = 0
    n_sifted:      int   = 0
    qber:          float = 0.0
    key_final:      str   = ""
    error_message: str   = ""

    # Champs de progression pour le client
    progress_pct:  float = 0.0   # 0–100
    phase_label:   str   = ""


def session_to_response(s: OrchestratorSession) -> SessionStatusResponse:

    progress = {
        SessionStatus.CREATED:      0.0,
        SessionStatus.INITIALIZING: 10.0,
        SessionStatus.SENDING:      40.0,
        SessionStatus.SIFTING:      80.0,
        SessionStatus.DONE:         100.0,
        SessionStatus.ABORTED:      0.0,
    }
    labels = {
        SessionStatus.CREATED:      "En attente",
        SessionStatus.INITIALIZING: "Initialisation réseau quantique",
        SessionStatus.SENDING:      "Transmission des qubits",
        SessionStatus.SIFTING:      "Sifting & calcul QBER",
        SessionStatus.DONE:         "Clé générée",
        SessionStatus.ABORTED:      "Session abandonnée",
    }

    return SessionStatusResponse(
        session_id=s.session_id,
        status=s.status,
        elapsed_s=round(s.elapsed_s, 2),
        n_qubits=s.n_qubits,
        n_delivered=s.n_delivered,
        n_sifted=s.n_sifted,
        qber=s.qber,
        key_final=s.key_final,
        error_message=s.error_message,
        progress_pct=progress.get(s.status, 0.0),
        phase_label=labels.get(s.status, ""),
    )


def new_session_id() -> str:
    return str(uuid.uuid4())
