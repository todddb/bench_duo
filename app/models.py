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


class Conversation(BaseModel):
    __tablename__ = "conversations"

    title = db.Column(db.String(255), nullable=False, default="Untitled Conversation")
