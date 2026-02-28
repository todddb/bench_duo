from datetime import datetime

from .extensions import db


class BaseModel(db.Model):
    __abstract__ = True

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Model(BaseModel):
    __tablename__ = "models"

    name = db.Column(db.String(255), unique=True, nullable=False)
    host = db.Column(db.String(255), nullable=False)
    port = db.Column(db.Integer, nullable=False)
    backend = db.Column(db.String(64), nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(32), nullable=False, default="unknown")

    agents = db.relationship("Agent", back_populates="model", cascade="all, delete-orphan")


class Agent(BaseModel):
    __tablename__ = "agents"

    name = db.Column(db.String(255), unique=True, nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("models.id"), nullable=False)
    system_prompt = db.Column(db.Text, nullable=False)
    max_tokens = db.Column(db.Integer, nullable=False, default=256)
    temperature = db.Column(db.Float, nullable=False, default=0.2)
    status = db.Column(db.String(32), nullable=False, default="ready")

    model = db.relationship("Model", back_populates="agents")
    conversation_slots_a = db.relationship(
        "Conversation",
        back_populates="agent1",
        foreign_keys="Conversation.agent1_id",
    )
    conversation_slots_b = db.relationship(
        "Conversation",
        back_populates="agent2",
        foreign_keys="Conversation.agent2_id",
    )
    messages = db.relationship("Message", back_populates="agent")


class Conversation(BaseModel):
    __tablename__ = "conversations"

    agent1_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    agent2_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    ttl = db.Column(db.Integer, nullable=False, default=10)
    finished_at = db.Column(db.DateTime, nullable=True)
    random_seed = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(32), nullable=False, default="pending")

    agent1 = db.relationship("Agent", foreign_keys=[agent1_id], back_populates="conversation_slots_a")
    agent2 = db.relationship("Agent", foreign_keys=[agent2_id], back_populates="conversation_slots_b")
    messages = db.relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    evaluation_jobs = db.relationship("EvaluationJob", back_populates="conversation")


class Message(BaseModel):
    __tablename__ = "messages"

    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False)
    sender_role = db.Column(db.String(32), nullable=False)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=True)
    content = db.Column(db.Text, nullable=False)
    tokens = db.Column(db.Integer, nullable=True)
    raw_response = db.Column(db.JSON, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    conversation = db.relationship("Conversation", back_populates="messages")
    agent = db.relationship("Agent", back_populates="messages")


class BatchJob(BaseModel):
    __tablename__ = "batch_jobs"

    agent1_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    agent2_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    num_runs = db.Column(db.Integer, nullable=False, default=1)
    completed_runs = db.Column(db.Integer, nullable=False, default=0)
    ttl = db.Column(db.Integer, nullable=False, default=10)
    seed = db.Column(db.Integer, nullable=True)
    cancel_requested = db.Column(db.Boolean, nullable=False, default=False)
    total_elapsed_seconds = db.Column(db.Float, nullable=False, default=0.0)
    total_tokens = db.Column(db.Integer, nullable=False, default=0)
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(32), nullable=False, default="queued")
    summary = db.Column(db.JSON, nullable=True)

    agent1 = db.relationship("Agent", foreign_keys=[agent1_id])
    agent2 = db.relationship("Agent", foreign_keys=[agent2_id])
    evaluation_jobs = db.relationship("EvaluationJob", back_populates="batch_job")


class EvaluationJob(BaseModel):
    __tablename__ = "evaluation_jobs"

    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("batch_jobs.id"), nullable=True)
    main_model_id = db.Column(db.Integer, db.ForeignKey("models.id"), nullable=False)
    judge_model_ids = db.Column(db.JSON, nullable=False, default=list)
    results = db.Column(db.JSON, nullable=True)
    report = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(32), nullable=False, default="pending")

    conversation = db.relationship("Conversation", back_populates="evaluation_jobs")
    batch_job = db.relationship("BatchJob", back_populates="evaluation_jobs")
    main_model = db.relationship("Model")


class ConnectorLog(BaseModel):
    __tablename__ = "connector_logs"

    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    connector_type = db.Column(db.String(64), nullable=False)
    request = db.Column(db.JSON, nullable=True)
    response = db.Column(db.JSON, nullable=True)
    success = db.Column(db.Boolean, nullable=False, default=True)
    error = db.Column(db.Text, nullable=True)
