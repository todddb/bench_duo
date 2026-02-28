#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from app import create_app
from app.extensions import db
from app.models import BatchJob, Conversation


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge old batch jobs and conversations")
    parser.add_argument("--days", type=int, default=30, help="Delete rows older than this many days")
    args = parser.parse_args()

    if args.days < 0:
        raise SystemExit("--days must be >= 0")

    cutoff = datetime.utcnow() - timedelta(days=args.days)
    app = create_app()
    with app.app_context():
        deleted_batch = BatchJob.query.filter(BatchJob.created_at < cutoff).delete(synchronize_session=False)
        deleted_conversations = Conversation.query.filter(Conversation.created_at < cutoff).delete(synchronize_session=False)
        db.session.commit()

    print(
        f"Purged {deleted_batch} batch job(s) and {deleted_conversations} conversation(s) older than {args.days} day(s)."
    )


if __name__ == "__main__":
    main()
