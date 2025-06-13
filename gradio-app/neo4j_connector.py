from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, Neo4jError

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except CypherSyntaxError as e:
            return {"error": f"Cypher syntax error: {e}"}
        except Neo4jError as e:
            return {"error": f"Neo4j error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

