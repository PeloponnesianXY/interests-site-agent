FROM python:3.12-slim

WORKDIR /app

COPY requirements-chainlit.txt .

RUN pip install --no-cache-dir -r requirements-chainlit.txt

COPY . .

CMD ["python", "-m", "chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
