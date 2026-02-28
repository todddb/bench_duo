from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    INSTANCE_DIR = BASE_DIR / "instance"
    INSTANCE_DIR.mkdir(exist_ok=True)

    SECRET_KEY = "dev-secret-key-change-me"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{INSTANCE_DIR / 'bench_duo.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
