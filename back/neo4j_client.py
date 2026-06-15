# neo4j_client.py

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_uri")
USER     = os.getenv("NEO4J_username")
PASSWORD = os.getenv("NEO4J_password")

# this driver will be imported and used by all other files
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def test_connection():
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j Aura!")
    except Exception as e:
        print(f" Connection failed: {e}")


if __name__ == "__main__":
    test_connection()