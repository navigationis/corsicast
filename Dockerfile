FROM python:3.12.1-slim as stage-1
COPY atm.py scripts/patgen.py /opt/corsicast/
RUN apt-get update; \
    apt-get install -y gcc
RUN python3 -m venv /opt/corsicast/.venv && \
    /opt/corsicast/.venv/bin/pip install numpy scipy MCEq

FROM python:3.12.1-slim as stage-2
RUN mkdir /mnt/data
WORKDIR /mnt/data
VOLUME /mnt/data
COPY --from=stage-1 /opt/corsicast/ /opt/corsicast/
ENV PYTHONPATH /opt/corsicast
# This import should force MCEq to pull down the hadronic data .h5 file
RUN echo "import MCEq.core" | /opt/corsicast/.venv/bin/python
ENTRYPOINT ["/opt/corsicast/.venv/bin/python3", "/opt/corsicast/patgen.py", "-p", "MDK", \
    "-t", "0", "10", "20", "30", "40", "50", "60", "70", "80", "90", \
    "-H", "2500", "1000", "500", "--"]
