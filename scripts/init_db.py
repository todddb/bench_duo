from app import create_app
from app.extensions import db


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()
        print("Database initialized.")


if __name__ == "__main__":
    main()
