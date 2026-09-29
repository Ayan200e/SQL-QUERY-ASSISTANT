import asyncio
import sqlite3 as sql

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Initialize client
client = genai.Client(api_key="my-api-key") #I have removed the key for security purposes (and also Github was not allowing)


def setup_database() -> sql.Connection:
    """Sets up in-memory SQLite database ONCE at startup."""
    conn = sql.connect(":memory:")  
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS orders")

    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            price REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        )
    """)

    product_data = [
        (101, "Mechanical Keyboard", "Electronic", 85.00),
        (102, "Wired Earphones", "Electronic", 40.00),
        (103, "Flowered theme Journal", "Stationary", 100.00),
        (104, "Color Pens", "Stationary", 50.00),
        (105, "Mouse", "Electronic", 35.00),
        (106, "Over-ear headphones", "Electronic", 90.00),
        (107, "Sunflower Lamp", "Electronic", 25.00),
    ]

    orders = [
        (1, 5001, 101, 1, "2026-09-01"),
        (2, 5002, 102, 2, "2026-09-02"),
        (3, 5003, 102, 1, "2026-09-23"),
        (4, 5001, 103, 1, "2026-09-17"),
        (5, 5004, 105, 2, "2026-09-06"),
        (6, 5002, 104, 1, "2026-09-30"),
        (7, 5001, 107, 1, "2026-09-11"),
        (8, 5005, 106, 3, "2026-09-28"),
    ]

    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", product_data)
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)
    conn.commit()
    return conn


class SQLResponse(BaseModel):
    explanation: str = Field(description="A brief expl. of how the query maps to users' requests.")
    sql_query: str = Field(description="The executable, raw query strings without markdown code blocks.")


async def generate_query(user_question: str) -> SQLResponse:
    """Async API call to Gemini using client.aio"""
    db_schema_context = """
        You are an expert AI Analyst. Your job is to convert natural language questions into valid SQLite code.

        Database Schema:
        - Table : products
          Columns: product_id(int , primary key) , product_name(text) , category(text) , price(real)
        - Table : orders
          Columns: order_id(int , primary key) , customer_id(int) , product_id(int , foreign key) , quantity(int) , order_date(text, yyyy-mm-dd)

        RULES:-
        1. Only use the tables and columns given above.
        2. Respond with valid SQLite Code.
        3. Do not include markdown code formatting (like ```sql) inside the query field.
    """

    # Use client.aio for non-blocking asynchronous calls
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"Convert this question into an SQL Query : {user_question}",
        config=types.GenerateContentConfig(
            system_instruction=db_schema_context,
            response_mime_type="application/json",
            response_schema=SQLResponse,
            temperature=0.1,
        ),
    )

    return SQLResponse.model_validate_json(response.text)


async def main():
    # 1. Initialize the DB ONCE before entering the interactive loop
    db_conn = setup_database()
    cur = db_conn.cursor()

    try:
        while True:
            # Use asyncio.to_thread for non-blocking console input
            user_prompt = await asyncio.to_thread(
                input, "Your Question (or type 'stop' to exit): "
            )
            user_prompt = user_prompt.strip()

            if user_prompt.lower() == "stop":
                print("Exiting...")
                break

            if not user_prompt:
                continue

            print(f"\nUser Question: '{user_prompt}'\n")

            # 2. Async API Call
            result = await generate_query(user_prompt)
            print(f"Explanation: {result.explanation}")
            print(f"Generated Query:\n{result.sql_query}\n")

            # 3. Execute query on existing connection
            try:
                cur.execute(result.sql_query)
                rows = cur.fetchall()

                print("Execution Results:")
                for row in rows:
                    print(row)
                print()

            except Exception as e:
                print(f"Execution failed: {e}\n")

    finally:
        db_conn.close()


if __name__ == "__main__":
    # Run the async loop
    asyncio.run(main())