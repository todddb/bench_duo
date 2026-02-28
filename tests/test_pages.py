import unittest

from app import create_app
from app.extensions import db


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class PagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(TestConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_pages_return_html(self) -> None:
        for path in ["/", "/chat", "/batch", "/evaluation"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Bench Duo", response.data)


if __name__ == "__main__":
    unittest.main()
