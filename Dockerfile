FROM python:3.13.3-slim
LABEL org.opencontainers.image.source="https://github.com/robinmarechal/dynatrace_exporter"

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app
RUN pip install --no-cache-dir -r requirements.txt

COPY dynatrace_exporter.py /usr/src/app

EXPOSE 9126
ENV LISTEN_ADDRESS=:9126

ENTRYPOINT [ "./dynatrace_exporter.py" ]
