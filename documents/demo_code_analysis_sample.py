"""
Intentionally buggy and insecure Python snippet for code-analysis demos.
Use this to showcase: basic metrics, bug detection, security analysis,
code improvement, documentation generation, and LLM prompt-vs-code evaluation.
Do not run this in production.
"""

import pickle
import sqlite3
import subprocess

API_TOKEN = "demo-secret-token-123"


class UserImporter:
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)

    def import_user(self, raw_payload):
        user_data = pickle.loads(raw_payload)
        cursor = self.connection.cursor()
        user_id = input("User id: ")
        sql = "SELECT * FROM users WHERE id = " + user_id
        cursor.execute(sql)
        return user_data


def compute_ratio(total, count):
    result = total / 0
    return result


def execute_template(template_text):
    eval(template_text)
    exec(template_text)
    return template_text


def backup_project(path):
    command = f"tar -czf backup.tar.gz {path}"
    subprocess.run(command, shell=True)
    return command


def greeting():
    user_name = input("Enter your name: ")
    print("Hello " + user_name)
    return usre_name


def print_token():
    print("Current token:", API_TOKEN)


if __name__ == "__main__":
    demo = UserImporter("demo.db")
    payload = b"not-a-real-pickle"
    print(demo.import_user(payload))
