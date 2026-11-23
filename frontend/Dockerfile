# Container for the Hugging Face Space (sdk: docker). HF runs containers as user 1000.
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user app.py .
ENV PATH="/home/user/.local/bin:$PATH"
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
