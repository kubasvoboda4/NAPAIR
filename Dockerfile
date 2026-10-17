FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The scripts use ./cdf_files, ./data/curated and ./parameter-boundaries.csv
# as defaults, so the working directory must stay /app.
COPY *.py *.csv ./
COPY cdf_files ./cdf_files
COPY data/curated ./data/curated
COPY data/curated.npz ./data/curated.npz

ENV PYTHONUNBUFFERED=1
# Mount your structures/output at /work and use absolute paths, e.g. --out /work/out
VOLUME /work
ENTRYPOINT ["python"]
CMD ["assign_in_structure.py", "--help"]
