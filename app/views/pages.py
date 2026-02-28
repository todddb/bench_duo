from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/")
def setup_page():
    return render_template("setup.html", page_name="setup")


@pages_bp.get("/chat")
def chat_page():
    return render_template("chat.html", page_name="chat")


@pages_bp.get("/batch")
def batch_page():
    return render_template("batch.html", page_name="batch")


@pages_bp.get("/evaluation")
def evaluation_page():
    return render_template("evaluation.html", page_name="evaluation")
